import unittest
import numpy as np
from complex_evaluate import pixel_sums,reduced,sobel,psnr,safe_value

class MetricTests(unittest.TestCase):
    def test_masked_rgb_counts_and_known_region(self):
        gt=np.zeros((8,8,3));damaged=gt.copy();pred=gt.copy()
        m=np.zeros((8,8),bool);m[2:4,2:4]=True
        pred[m]=[1,0,0];pred[0,0]=[0,1,0]
        s=pixel_sums(pred,gt,damaged,m,m.copy());r=reduced(s)
        self.assertEqual(s['masked_pixels'],4)
        self.assertAlmostEqual(r['masked_mae'],1/3)
        self.assertAlmostEqual(r['masked_mse'],1/3)
        self.assertAlmostEqual(r['known_mae'],1/(60*3))

    def test_empty_and_zero_are_distinct(self):
        x=np.zeros((8,8,3));m=np.zeros((8,8),bool)
        empty=reduced(pixel_sums(x,x,x,m,m))
        self.assertEqual(empty['masked_mae'],'N/A')
        self.assertEqual(empty['structure_sobel_l1'],'N/A')
        self.assertEqual(empty['known_mae'],0)
        exact=reduced(pixel_sums(x,x,x,~m,~m))
        self.assertEqual(exact['masked_psnr'],'Infinity')
        self.assertEqual(exact['known_mae'],'N/A')
        self.assertEqual(safe_value(float('inf')-float('inf')),'N/A')

    def test_sobel_normalization_and_reflect(self):
        ramp=np.broadcast_to(np.arange(8)[None,:,None]/8,(8,8,3)).copy()
        gx,gy=sobel(ramp)
        np.testing.assert_allclose(gx[:,1:-1],1/8,atol=1e-15)
        np.testing.assert_allclose(gx[:,[0,-1]],0,atol=1e-15)
        np.testing.assert_allclose(gy,0,atol=1e-15)
        c=np.ones((8,8,3))*.7
        for gradient in sobel(c):np.testing.assert_allclose(gradient,0,atol=1e-15)

    def test_pixel_weighted_building_then_psnr(self):
        gt=np.zeros((8,8,3));p=np.ones_like(gt)*.2
        one=np.zeros((8,8),bool);one[0,0]=True
        many=np.zeros_like(one);many[:2,:2]=True
        a=pixel_sums(p,gt,gt,one,one)
        b=pixel_sums(p*2,gt,gt,many,many)
        result=reduced({key:a[key]+b[key] for key in a})
        self.assertAlmostEqual(result['masked_mae'],(.2+4*.4)/5)
        self.assertAlmostEqual(result['masked_mse'],(.04+4*.16)/5)
        self.assertAlmostEqual(result['masked_psnr'],psnr((.04+4*.16)/5))

if __name__=='__main__':unittest.main(verbosity=2)
