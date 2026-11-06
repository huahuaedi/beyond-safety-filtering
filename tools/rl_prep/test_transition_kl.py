import ast
import math
import unittest
from pathlib import Path
import torch
from transition_kl import equal_variance_transition_kl as kl

class TransitionKLTest(unittest.TestCase):
    def test_matches_torch_distribution_and_gradient(self):
        p=torch.tensor([[[[1., -2.]]]], requires_grad=True)
        r=torch.zeros_like(p, requires_grad=True)
        expected=torch.distributions.kl_divergence(torch.distributions.Normal(p, .5),torch.distributions.Normal(r,.5)).mean()
        result=kl(p,r,.5)
        self.assertTrue(torch.allclose(result,expected))
        result.backward()
        self.assertTrue(torch.allclose(p.grad, p.detach()/(.5**2*p.numel())))
        self.assertIsNone(r.grad)
    def test_no_cancellation_and_identity(self):
        p=torch.tensor([1.,-1.]); r=torch.zeros(2)
        self.assertEqual(kl(p,r,1.).item(),.5)
        self.assertEqual(kl(p,p,1.).item(),0.)
    def test_normalization_and_noise_scale(self):
        p=torch.ones(2); r=torch.zeros(2)
        self.assertEqual(kl(p.repeat(5),r.repeat(5),1.),kl(p,r,1.))
        self.assertEqual(kl(p,r,2.)*4,kl(p,r,1.))
    def test_invalid_inputs(self):
        for sigma in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError): kl(torch.ones(1),torch.ones(1),sigma)
        with self.assertRaises(ValueError): kl(torch.ones(1),torch.ones(2),1.)
    def test_real_recompute_optional_means(self):
        # Exercise the production function without importing/loading the VLM.
        tree=ast.parse(Path(__file__).with_name('rl_stage1_prototype.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='recompute_log_prob')
        fn.returns=None
        for arg in fn.args.args: arg.annotation=None
        class Obj: pass
        policy=Obj(); policy.vlm=Obj(); policy.vlm.model=Obj()
        policy.vlm.model.embed_tokens=lambda ids: torch.zeros(1,2,3)
        heads=Obj(); heads.state_projector=lambda x:x
        weight=torch.tensor(2.,requires_grad=True)
        ns={'torch':torch,'np':__import__('numpy'),'denoise_step_velocity':lambda *args: weight*torch.ones_like(args[3])}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'production_function','exec'),ns)
        args=(policy,heads,torch.ones(1,2,dtype=torch.long),None,torch.zeros(1,1),torch.zeros(1,1),[{'z':torch.zeros(1,2,3),'z_next':torch.ones(1,2,3),'t':1.,'_dt':-.1}],.2,2)
        legacy=ns['recompute_log_prob'](*args)
        logp,means=ns['recompute_log_prob'](*args,return_means=True)
        self.assertTrue(torch.equal(legacy,logp))
        self.assertEqual(tuple(means.shape),(1,1,2,3))
        kl(means,torch.zeros_like(means),.2).backward()
        self.assertGreater(weight.grad.item(),0.)

if __name__=='__main__': unittest.main()
