"""Small CPU tests of the real trainer loss, without loading pretrained weights."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import train_offline_verified_policy as trainer


class OfflineFlowTests(unittest.TestCase):
    def setUp(self):
        self.emb=nn.Embedding(3,4)
        nn.init.zeros_(self.emb.weight)
        self.heads=SimpleNamespace(state_projector=nn.Linear(13,4),advantage_condition_embedding=self.emb)
        self.policy=SimpleNamespace(vlm=SimpleNamespace(model=SimpleNamespace(embed_tokens=nn.Embedding(10,4))),
            rf=SimpleNamespace(sample_method='uniform',s=1.,add_noise=lambda a,n,t:(1-t)*a+t*n))
        self.sample={'input_ids':torch.tensor([1,2]),'attention_mask':torch.ones(2,dtype=torch.long),
            'pre_action':torch.zeros(3,3),'cmd':torch.zeros(4)}
        self.target=np.zeros((8,3),np.float32)

    def velocity(self,heads,vlm,state,z,*args):
        return state[...,:3].expand_as(z)

    def loss(self,label,seed=123):
        with patch.object(trainer.core,'DEVICE','cpu'),patch.object(trainer.core,'DTYPE',torch.float32),patch.object(trainer.core,'denoise_step_velocity',side_effect=self.velocity):
            return trainer.flow_loss(self.policy,self.heads,self.sample,self.target,label,seed)

    def test_zero_condition_preserves_base_predictions(self):
        self.assertEqual(float(self.loss(0)),float(self.loss(2)))

    def test_loss_reproducible_and_does_not_advance_cpu_rng(self):
        torch.manual_seed(87);before=torch.get_rng_state().clone()
        a=self.loss(2);self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(float(a),float(self.loss(2)))

    def test_conditioned_gradient_reaches_only_selected_row(self):
        self.loss(2).backward()
        self.assertGreater(float(self.emb.weight.grad[2].abs().sum()),0.)
        self.assertEqual(float(self.emb.weight.grad[:2].abs().sum()),0.)

    def test_unconditional_gradient_reaches_unconditional_row(self):
        self.loss(0).backward()
        self.assertGreater(float(self.emb.weight.grad[0].abs().sum()),0.)
        self.assertEqual(float(self.emb.weight.grad[1:].abs().sum()),0.)

if __name__=='__main__': unittest.main()
