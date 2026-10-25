import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import complex_common as common
from run_eval import jobs


class InferenceContractTests(unittest.TestCase):
    def test_conditions_only_read_damaged_and_exact(self):
        damaged=np.full((512,512,3),73,dtype=np.uint8)
        exact=np.zeros((512,512),dtype=np.uint8);exact[20:24,30:34]=255
        seen=[]
        def source(cs,role,frame):
            seen.append((role,frame))
            if role=='damaged':return Image.fromarray(damaged)
            if role=='exact':return Image.fromarray(exact)
            raise AssertionError('GT/structure must not be opened during inference')
        with patch.object(common,'source',source):
            result=common.conditions('CS09',count=2)
        self.assertEqual(seen,[('damaged',0),('exact',0),('damaged',1),('exact',1)])
        image=np.asarray(result['conditioning'][0]);mask=np.asarray(result['model_mask'][0])
        np.testing.assert_array_equal(image[exact==0],damaged[exact==0])
        self.assertTrue((image[exact==255]==127).all())
        self.assertEqual(set(np.unique(mask)),{0,255})
        self.assertNotIn('gt',result)

    def test_black_exact_is_preserved(self):
        def source(cs,role,frame):
            return Image.new('RGB',(512,512),(10,20,30)) if role=='damaged' else Image.new('L',(512,512),0)
        with patch.object(common,'source',source):result=common.conditions('CS09',count=1)
        self.assertEqual(result['conditioning'][0].getpixel((0,0)),(10,20,30))
        self.assertFalse(np.asarray(result['model_mask'][0]).any())

    def test_external_weights_do_not_depend_on_new_training(self):
        c={'weights':{'C':{'path':'/prior/checkpoints/C/step300.safetensors'}}}
        with patch.object(common,'config',return_value=c):
            self.assertEqual(str(common.lora_path('C')),c['weights']['C']['path'])

    def test_queue_has_exactly_sixty_inferences_and_no_training(self):
        queue=jobs();inference=[j for j in queue if j[0].startswith('infer_')]
        self.assertEqual(len(inference),60)
        self.assertEqual(len({j[0] for j in inference}),60)
        self.assertEqual([j[0] for j in inference[:5]],[f'infer_{m}_CS09' for m in 'BLCUS'])
        self.assertFalse(any('train' in j[0] or any('train' in a for a in j[1]) for j in queue))


if __name__=='__main__':unittest.main()
