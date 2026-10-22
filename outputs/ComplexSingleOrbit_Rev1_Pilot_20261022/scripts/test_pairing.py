import unittest
from unittest.mock import patch
from types import SimpleNamespace
from complex_common import *
from complex_train import weighted_error, flow_loss
import torch

class PairingTests(unittest.TestCase):
    def test_mapping_fine_line_and_local_temporal_groups(self):
        x=np.zeros((17,512,512),dtype=bool)
        for f in range(17): x[f,f*8,0]=True
        grid=loss_grid(x)
        self.assertEqual(grid.shape,(5,64,64))
        self.assertEqual(int(grid.sum()),17)
        for t,g in enumerate(TEMPORAL_GROUPS):
            self.assertEqual(np.flatnonzero(grid[t,:,0]).tolist(),g)

    def test_weight_and_empty_fallback(self):
        torch.manual_seed(1)
        p=torch.randn(1,16,5,8,8);t=torch.randn_like(p)
        m=torch.zeros(1,5,8,8);m[:,:,:4]=1
        e=torch.zeros_like(m);e[:,:,2,:2]=1
        raw=torch.nn.functional.mse_loss(p,t)
        for method in ('C','U','S'):
            self.assertTrue(torch.equal(weighted_error(p,t,m,e,method,strength=0),raw))
            self.assertTrue(torch.equal(weighted_error(p,t,m,e*0,method),raw))
            self.assertTrue(torch.equal(weighted_error(p,t,m*0,e*0,method),raw))
        self.assertAlmostEqual(float((2*e.sum()/m.sum()*m).sum()),float((2*e).sum()),places=5)
        # A localized error in E is deliberately emphasized by S more than U.
        q=e[:,None].expand_as(p)
        self.assertGreater(float(weighted_error(q,p*0,m,e,'S')),float(weighted_error(q,p*0,m,e,'U')))

    def test_original_flowmatch_equivalence(self):
        from diffsynth.diffusion import FlowMatchSFTLoss
        from diffsynth.diffusion.flow_match import FlowMatchScheduler
        scheduler=FlowMatchScheduler('Wan');scheduler.set_timesteps(1000,training=True)
        pipe=SimpleNamespace(device='cpu',torch_dtype=torch.bfloat16,scheduler=scheduler,in_iteration_models=[],
                             model_fn=lambda **kwargs:kwargs['latents']*.7)
        x=torch.randn(1,16,5,8,8,dtype=torch.bfloat16)
        row={'timestep_index':543,'noise_seed':99}
        noise=torch.randn(x.shape,dtype=x.dtype,generator=torch.Generator().manual_seed(99))
        with patch('torch.randint',return_value=torch.tensor([543])),patch('torch.randn_like',return_value=noise):
            original=FlowMatchSFTLoss(pipe,input_latents=x)
        m=torch.ones(1,5,8,8);e=m.clone()
        result,_,_=flow_loss(pipe,{'input_latents':x},m,e,'C',row)
        self.assertTrue(torch.equal(original,result))

    def test_source_condition_polarity_and_sampling(self):
        from vace_p01 import check_loader_contract
        torch.set_num_threads(4)
        for cs in ('CS06','CS09'):
            x=conditions(cs,0,17)
            check_loader_contract(x['conditioning'],x['model_mask'],num_frames=17)
            self.assertNotIn('gt',x)
        table=read_csv(RUN/'shared_sampling_table.csv')
        starts=read_json(RUN/'checks/eligible_windows.json')
        self.assertEqual([int(r['update']) for r in table],list(range(1,301)))
        for r in table:
            self.assertIn(r['cs'],TRAIN)
            self.assertIn(int(r['start']),starts[r['cs']])
            self.assertLessEqual(int(r['start'])+16,79)

if __name__=='__main__': unittest.main(verbosity=2)
