"""CPU-only regression checks for strict, single-read model SHA256 verification."""

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vace_p01


class ModelHashVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="building_wan_model_hash_cpu_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "checkpoint.safetensors"
        self.content = b"approved model payload fixture"
        self.path.write_bytes(self.content)
        self.expected = hashlib.sha256(self.content).hexdigest()

    def test_approved_hash_is_read_once_without_modifying_the_source(self):
        before = self.path.stat()
        real_sha256 = vace_p01.sha256
        with patch.object(vace_p01, "sha256", wraps=real_sha256) as checksum:
            actual = vace_p01.verify_file_sha256(self.path, {self.expected}, "checkpoint")
        checksum.assert_called_once_with(self.path)
        self.assertEqual(actual, self.expected)
        self.assertEqual(self.path.read_bytes(), self.content)
        after = self.path.stat()
        for attribute in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
            self.assertEqual(getattr(before, attribute), getattr(after, attribute), attribute)

    def test_wrong_hash_reports_expected_actual_path_and_bytes_without_retry(self):
        approved = "0" * 64
        real_sha256 = vace_p01.sha256
        with patch.object(vace_p01, "sha256", wraps=real_sha256) as checksum:
            with self.assertRaises(ValueError) as failure:
                vace_p01.verify_file_sha256(self.path, {approved}, "checkpoint")
        checksum.assert_called_once_with(self.path)
        message = str(failure.exception)
        for expected_text in ("checkpoint", f"expected={approved}", f"actual={self.expected}",
                              f"path={self.path}", f"bytes={len(self.content)}", "without retry"):
            self.assertIn(expected_text, message)
        self.assertEqual(self.path.read_bytes(), self.content)

    def test_multiple_approved_hashes_accept_only_the_matching_digest(self):
        result = vace_p01.verify_file_sha256(self.path, {"0" * 64, self.expected}, "checkpoint")
        self.assertEqual(result, self.expected)

    def test_empty_approved_set_is_refused(self):
        with self.assertRaisesRegex(ValueError, "approved|expected=|SHA256"):
            vace_p01.verify_file_sha256(self.path, set(), "checkpoint")
        self.assertEqual(self.path.read_bytes(), self.content)

    def test_atomic_replacement_during_hash_is_refused_even_when_bytes_match(self):
        before = self.path.stat()
        replacement = self.root / "replacement.safetensors"
        replacement.write_bytes(self.content)
        real_sha256 = vace_p01.sha256

        def hash_and_replace(path):
            result = real_sha256(path)
            os.replace(replacement, path)
            return result

        with patch.object(vace_p01, "sha256", side_effect=hash_and_replace) as checksum:
            with self.assertRaisesRegex(ValueError, "changed during SHA256.*without retry"):
                vace_p01.verify_file_sha256(self.path, {self.expected}, "checkpoint")
        checksum.assert_called_once_with(self.path)
        self.assertNotEqual(self.path.stat().st_ino, before.st_ino)
        self.assertEqual(self.path.read_bytes(), self.content)

    def test_in_place_change_during_hash_is_refused_even_when_reported_hash_is_approved(self):
        before = self.path.stat()
        changed_content = b"X" * len(self.content)
        real_sha256 = vace_p01.sha256

        def hash_and_modify(path):
            result = real_sha256(path)
            Path(path).write_bytes(changed_content)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
            return result

        with patch.object(vace_p01, "sha256", side_effect=hash_and_modify) as checksum:
            with self.assertRaisesRegex(ValueError, "changed during SHA256.*without retry"):
                vace_p01.verify_file_sha256(self.path, {self.expected}, "checkpoint")
        checksum.assert_called_once_with(self.path)
        self.assertEqual(self.path.stat().st_ino, before.st_ino)
        self.assertEqual(self.path.read_bytes(), changed_content)

    def inspect_fixture(self, checkpoint_checksum):
        paths = {name: self.root / filename for name, filename in (
            ("checkpoint", "checkpoint.safetensors"),
            ("text_encoder", "text_encoder.safetensors"),
            ("vae", "vae.safetensors"),
        )}
        paths["text_encoder"].write_bytes(b"text encoder fixture")
        paths["vae"].write_bytes(b"VAE fixture")
        tokenizer = self.root / "tokenizer"
        tokenizer.mkdir()
        (tokenizer / "tokenizer.json").write_bytes(b"tokenizer fixture")
        config = dict(model=dict(id="fixture/vace", text_encoder=paths["text_encoder"].name,
                                 vae=paths["vae"].name, tokenizer=tokenizer.name))
        sources = dict(
            model_id="fixture/vace", huggingface_revision="frozen-fixture-revision",
            checkpoint=dict(default_local_path=self.path.name,
                            accepted_files=[dict(bytes=len(self.content), sha256=checkpoint_checksum)],
                            structure_md5="checkpoint_structure", tensor_count=2, vace_tensor_count=1),
            reused_components={
                name: dict(bytes=path.stat().st_size, sha256=vace_p01.sha256(path))
                for name, path in paths.items() if name != "checkpoint"
            },
        )
        sources["reused_components"]["tokenizer"] = dict(
            files_sha256={"tokenizer.json": vace_p01.sha256(tokenizer / "tokenizer.json")})
        headers = {
            paths["checkpoint"]: dict(bytes=len(self.content), tensors=2, vace_tensors=1,
                                      structure_md5="checkpoint_structure"),
            **{path: dict(bytes=path.stat().st_size, tensors=1, vace_tensors=0,
                          structure_md5=name + "_structure")
               for name, path in paths.items() if name != "checkpoint"},
        }
        registry = {
            "checkpoint_structure": ["wan_video_dit", "wan_video_vace"],
            "text_encoder_structure": ["wan_video_text_encoder"],
            "vae_structure": ["wan_video_vae"],
        }
        return config, sources, headers, registry

    def test_inspect_models_propagates_the_actual_checkpoint_digest_and_stops(self):
        approved = "0" * 64
        config, sources, headers, registry = self.inspect_fixture(approved)
        notices = []
        real_sha256 = vace_p01.sha256
        with patch.object(vace_p01, "safetensors_header", side_effect=lambda path: headers[path]), \
             patch.object(vace_p01, "registered_models", side_effect=lambda structure, root: registry[structure]), \
             patch.object(vace_p01, "sha256", wraps=real_sha256) as checksum:
            with self.assertRaises(ValueError) as failure:
                vace_p01.inspect_models(config, verify=True, project_root=self.root, sources=sources,
                                        notify=notices.append)
        checksum.assert_called_once_with(self.path)
        for expected_text in (f"expected={approved}", f"actual={self.expected}",
                              f"path={self.path}", f"bytes={len(self.content)}"):
            self.assertIn(expected_text, str(failure.exception))
        self.assertEqual(notices, [f"Reading full SHA256: checkpoint, {len(self.content):,} bytes"])

    def test_inspect_models_reports_verified_real_digests_without_loading_models(self):
        config, sources, headers, registry = self.inspect_fixture(self.expected)
        real_sha256 = vace_p01.sha256
        with patch.object(vace_p01, "safetensors_header", side_effect=lambda path: headers[path]), \
             patch.object(vace_p01, "registered_models", side_effect=lambda structure, root: registry[structure]), \
             patch.object(vace_p01, "sha256", wraps=real_sha256) as checksum:
            report = vace_p01.inspect_models(config, verify=True, project_root=self.root, sources=sources)
        self.assertTrue(report["checksums_verified"])
        self.assertTrue(report["ready_for_gpu"])
        self.assertEqual(report["files_sha256"]["checkpoint"], self.expected)
        self.assertEqual(checksum.call_count, 4)
        self.assertEqual(report["file_identities"]["checkpoint"]["bytes"], len(self.content))
        self.assertEqual(self.path.read_bytes(), self.content)


if __name__ == "__main__":
    unittest.main()
