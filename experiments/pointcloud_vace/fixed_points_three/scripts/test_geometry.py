"""CPU checks for gap coverage, abstention and preservation contracts."""
import unittest
import numpy as np
from candidates import generate, in_box


class FixedGeometryTests(unittest.TestCase):
    def test_two_sided_measured_wall_strips_bridge_a_gap(self):
        x,z=np.meshgrid(np.arange(-4.,4.01,.25),np.arange(-2.,2.01,.5))
        full=np.stack([x.ravel(),np.zeros(x.size),z.ravel()],axis=1)
        center=np.zeros(3);axes=np.eye(3);half=np.array([1.,.2,1.])
        observed=full[~in_box(full,center,axes,half)]
        colors=np.full(observed.shape,93,dtype=np.uint8)
        before=observed.copy();before_colors=colors.copy()
        xyz,rgb,keep,audit=generate(observed,colors,center,axes,half)
        self.assertGreater(len(keep),20)
        self.assertTrue(in_box(xyz,center,axes,half).all())
        self.assertLess(abs(xyz[keep,1]).max(),1.e-8)
        self.assertTrue(np.array_equal(observed,before))
        self.assertTrue(np.array_equal(colors,before_colors))
        self.assertTrue((rgb[keep]==93).all())
        from scipy.spatial import cKDTree
        truth=full[in_box(full,center,axes,half)]
        self.assertLess(float(np.quantile(cKDTree(xyz[keep]).query(truth)[0],.9)),.26)

    def test_no_local_surface_evidence_abstains(self):
        rng=np.random.default_rng(6)
        points=rng.normal(size=(40,3))+50
        xyz,rgb,keep,audit=generate(points,np.zeros((40,3),np.uint8),np.zeros(3),np.eye(3),np.ones(3))
        self.assertEqual(len(xyz),0);self.assertEqual(len(keep),0)
        self.assertTrue(audit["abstained"])


if __name__ == "__main__":
    unittest.main()
