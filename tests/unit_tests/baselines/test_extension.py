"""Baseline preparation, native capture, ownership and phase contract regression."""
from types import SimpleNamespace
from unittest.mock import patch
import os
import torch
import pytest
from megatron.baseline_sampling import SourceSampling, round_robin, read_source_sampling
from megatron.baseline_sites import Observation, RouterSelection, ExpertOutput, CaptureBase
from megatron.baseline_weights import qk_projection_ranges, assign_weight_fragments


def test_selector_configuration(tmp_path):
    config = tmp_path/'hooks.yaml'
    config.write_text('hooks:\n  moe_packed_weighted_output:\n    source_sampling:\n      function: dmi_megatron_integration.hooks.source_sampling.round_robin\n      args: {count: 2, offset: 1}\n')
    p = read_source_sampling(str(config), ep_enabled=True)
    assert p.select(1,4)==(1,2)
    assert p.select(2,4)==(0,3)
    assert p.select(2,4)==p.select(2,4)
    assert read_source_sampling(None,ep_enabled=True) is None
    assert SourceSampling(p.function, {'count':4}).select(4,4)==(0,1,2,3)
    with pytest.raises(ValueError):
        SourceSampling(p.function, {'count':5}).select(1,4)
    config.write_text('hooks:\n  hidden_states:\n    source_sampling: {function: unavailable.plugin, args: {}}\n')
    with pytest.warns(UserWarning):
        assert read_source_sampling(str(config),ep_enabled=True) is None


def test_custom_selector(tmp_path, monkeypatch):
    (tmp_path/'baseline_test_selector.py').write_text('def choose(iteration, num_sources, stride=2):\n return range(iteration % 2, num_sources, stride)\ndef bad(iteration, num_sources):\n return [0,0]\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    assert SourceSampling('baseline_test_selector.choose',{'stride':2}).select(1,4)==(0,2)
    assert SourceSampling('baseline_test_selector.choose',{'stride':2}).select(2,4)==(1,3)
    with pytest.raises(ValueError):
        SourceSampling('baseline_test_selector.bad',{}).select(1,4)


def test_actual_routes_not_logits():
    prep=RouterSelection(0,2,{'selected_expert_ids','routing_weights'})
    result={}
    prep.ids.register_forward_hook(lambda m,a,o: result.update(ids=o.clone()))
    prep.weights.register_forward_hook(lambda m,a,o: result.update(weights=o.clone()))
    probs=torch.tensor([[.2,.8,0.],[0.,.1,.9]])
    routing=torch.tensor([[True,False,False],[False,True,True]])
    prep(probs,routing,2,1)
    assert torch.equal(result['ids'],torch.tensor([[[0,3],[1,2]]]))
    torch.testing.assert_close(result['weights'],torch.tensor([[[.2,0],[.1,.9]]]))
    prep.enabled=False
    prep(None,None,0,0)


@pytest.mark.parametrize('blocks',[1,3])
@pytest.mark.parametrize('count',[1,2,4])
def test_segment_preparation(blocks,count):
    counts=torch.arange(4*blocks).reshape(4,blocks)%3
    dispatcher=SimpleNamespace(tp_size=2,ep_size=2,tp_rank=0,num_local_experts=blocks,
                               local_expert_indices=list(range(blocks)),num_global_tokens_per_local_expert=counts)
    ctx=SimpleNamespace(iteration=1)
    policy=SourceSampling('megatron.baseline_sampling.round_robin',{'count':count,'offset':1})
    prep=ExpertOutput(0,dispatcher,policy,ctx)
    received=[]
    prep.payload.register_forward_hook(lambda m,a,o: received.append(o.clone()))
    x=torch.arange(int(counts.sum())*2).reshape(-1,2)
    prep(x)
    keep=policy.select(2,4)
    labels=torch.repeat_interleave(torch.arange(4).repeat(blocks),counts.T.reshape(-1))
    mask=torch.tensor([int(i) in keep for i in labels],dtype=torch.bool)
    assert torch.equal(received[0],x[mask])
    assert prep.payload.extra_metadata['selected_sources']==list(keep)
    prep.enabled=False
    prep(None)


@pytest.mark.parametrize('gated',[False,True])
def test_qk_layout_and_replica_dedup(gated):
    reports=[];raw={}; reference={}
    h,g,d,hidden=8,2,4,3
    group_rows=h//g*d*(2 if gated else 1)+2*d
    full=torch.arange(g*group_rows*hidden,dtype=torch.float32).reshape(-1,hidden)
    for proj in ('q','k'):
        parts=[]
        for group in range(g):
            start=group*group_rows+(0 if proj=='q' else h//g*d*(2 if gated else 1))
            width=h//g*d if proj=='q' else d
            parts.append(full[start:start+width])
        reference[proj]=torch.cat(parts).view(torch.uint8).flatten()
        for tp in range(4):
            shape,local,ranges=qk_projection_ranges(heads=h,groups=g,head_dim=d,hidden=hidden,
                tp_rank=tp,tp_size=4,element_size=4,projection=proj,attention_output_gate=gated)
            for dp in range(2):
                rank=dp*4+tp
                raw[rank]=full.chunk(4)[tp].view(torch.uint8).flatten()
                reports.append(dict(layer_no=0,act_name=proj,producer_rank=rank,shape=shape,dtype='float32',available=ranges,source_kind='replicated'))
    layouts=assign_weight_fragments(reports)
    for proj in ('q','k'):
        output=torch.empty_like(reference[proj]);covered=torch.zeros(output.numel(),dtype=torch.int)
        for layout in layouts:
            if layout['act_name']!=proj: continue
            for src,dst,size in layout['fragments']:
                output[dst:dst+size]=raw[layout['producer_rank']][src:src+size]
                covered[dst:dst+size]+=1
        assert torch.equal(output,reference[proj])
        assert torch.all(covered==1)


class TransformerLayer(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.layer_number=1
    def forward(self,x):
        if hasattr(self,'baseline_hidden_states'): self.baseline_hidden_states(x)
        return x*2

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.layer=TransformerLayer()
    def forward(self,x): return self.layer(x)


def test_native_values_and_disabled_phase():
    from megatron.baseline_capture import Capture
    coords=dict(tp_rank=0,tp_size=1,pp_rank=0,pp_size=1,dp_rank=0)
    with patch('megatron.baseline_sites.parallel_coordinates',return_value=coords):
        model=Model();mode=os.environ.get('BASELINE_CAPTURE_MODE','immediate')
        cap=Capture(model,selected=['hidden_states'],mode=mode)
    device='cuda' if os.environ.get('BASELINE_TEST_CUDA') else 'cpu'
    x=torch.arange(12,device=device).reshape(3,4).float()
    for step in range(2):
        cap.begin_iteration(step);cap.microbatch=0
        result=cap.forward(x+step)
        cap.end_iteration()
        torch.testing.assert_close(cap.records[-1]['tensor'],(x+step).cpu())
        torch.testing.assert_close(result,(x+step)*2)
    n=len(cap.records);cap.set_enabled(False);model(x)
    assert len(cap.records)==n
    cap.close()

class SelfAttention(torch.nn.Module):
    def __init__(self, gated=False):
        super().__init__()
        self.layer_number=1
        self.hidden_size_per_attention_head=2
        self.config=SimpleNamespace(num_query_groups=2,num_attention_heads=4,hidden_size=4,
                                    attention_output_gate=gated,fp8=None,fp4=None)
        self.linear_qkv=torch.nn.Linear(4,(8*(2 if gated else 1)+8),bias=False)

class GatedDeltaNet(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layer_number=2;self.in_proj=torch.nn.Linear(4,16,bias=False)

class WeightModel(torch.nn.Module):
    def __init__(self,gated=False):
        super().__init__();self.attention=SelfAttention(gated);self.linear_attention=GatedDeltaNet()
        self.layer=TransformerLayer();self.calls=0
    def forward(self,x):
        self.calls+=1
        return self.layer(x)


@pytest.mark.parametrize('gated',[False,True])
def test_native_weight_and_activation_groups(gated):
    from megatron.baseline_capture import Capture
    coords=dict(tp_rank=0,tp_size=1,pp_rank=0,pp_size=1,dp_rank=0)
    device='cuda' if os.environ.get('BASELINE_TEST_CUDA') else 'cpu'
    with patch('megatron.baseline_sites.parallel_coordinates',return_value=coords):
        model=WeightModel(gated).to(device)
        cap=Capture(model,selected=['hidden_states','qk_weights'],mode=os.environ.get('BASELINE_CAPTURE_MODE','immediate'))
    assert len(cap.weight_points)==2  # Linear-attention projections excluded.
    for iteration in range(2):
        cap.begin_iteration(iteration);cap.microbatch=0
        cap.forward(torch.ones(3,4,device=device))
        cap.capture_weights()
        # Native async consumes the packed snapshot, not mutated model storage.
        with torch.no_grad(): model.attention.linear_qkv.weight.add_(10)
        cap.end_iteration()
        rows=cap.records[-3:]
        assert {r['hook'] for r in rows}=={'hidden_states','query_projection_weight','key_projection_weight'}
        for row in rows:
            if 'weight_layout' not in row:continue
            layout=row['weight_layout'];actual=torch.empty(8*4 if row['hook'].startswith('query') else 4*4,dtype=torch.float32).view(torch.uint8)
            pos=0
            for _,dest,length in layout['fragments']:
                actual[dest:dest+length]=row['tensor'][pos:pos+length];pos+=length
            packed=(model.attention.linear_qkv.weight.detach()-10).reshape(2,-1,4)
            expected=(packed[:,:4] if row['hook'].startswith('query') else packed[:,8 if gated else 4:(8 if gated else 4)+2]).contiguous().flatten()
            torch.testing.assert_close(actual.view(torch.float32),expected.cpu(),rtol=0,atol=2e-6)
    assert model.calls==2
    cap.close()

class MoEAlltoAllTokenDispatcher:
    def __init__(self):
        self.tp_size=self.ep_size=1;self.tp_rank=0;self.num_local_experts=2
        self.local_expert_indices=[0,1]
        self.num_global_tokens_per_local_expert=torch.tensor([[4,4]])

class TopKRouter(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layer_number=1;self.topk=2
    def forward(self,x):
        logits=x[:,:2]
        if hasattr(self,'baseline_router_logits'): self.baseline_router_logits(logits)
        probs=torch.softmax(logits,dim=-1)
        if hasattr(self,'baseline_router_selection'):
            self.baseline_router_selection(probs,torch.ones_like(probs,dtype=torch.bool),len(x),1)
        return probs

class MoELayer(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layer_number=1;self.router=TopKRouter()
        self.token_dispatcher=MoEAlltoAllTokenDispatcher()
    def forward(self,x):
        self.router(x)
        if hasattr(self,'baseline_moe_inverse_map'):
            self.baseline_moe_inverse_map(torch.arange(len(x),device=x.device).repeat(2))
        if hasattr(self,'baseline_moe_packed_weighted_output'):
            self.baseline_moe_packed_weighted_output(x.repeat(2,1))
        return x


def test_native_route_inverse_output_order():
    from megatron.baseline_capture import Capture
    coords=dict(tp_rank=0,tp_size=1,pp_rank=0,pp_size=1,dp_rank=0)
    with patch('megatron.baseline_sites.parallel_coordinates',return_value=coords):
        model=MoELayer()
        cap=Capture(model,selected=['router_logits','selected_expert_ids','routing_weights',
            'moe_inverse_map','moe_packed_weighted_output'],mode=os.environ.get('BASELINE_CAPTURE_MODE','immediate'))
    device='cuda' if os.environ.get('BASELINE_TEST_CUDA') else 'cpu'
    x=torch.arange(16,device=device).float().reshape(4,4)
    cap.begin_iteration(0);cap.microbatch=0;cap.forward(x);cap.end_iteration()
    assert len(cap.records)==5
    result={r['hook']:r['tensor'] for r in cap.records}
    torch.testing.assert_close(result['moe_packed_weighted_output'],x.repeat(2,1).cpu())
    assert result['selected_expert_ids'].shape==(1,4,2)
    cap.close()


def test_validation_uses_its_own_schedule():
    from unittest.mock import MagicMock
    from megatron.baseline_workload_evaluation import WorkloadEvaluation
    e=WorkloadEvaluation.__new__(WorkloadEvaluation)
    e.active=False;e.training_start=None;e.capture_phase='validation'
    e.capture=MagicMock();e.capture.records=[]
    with patch('megatron.core.num_microbatches_calculator.get_num_microbatches',return_value=2):
        e.begin_iteration(0)
        assert e.microbatches==2 and not e.active
        e.begin_iteration(0,phase='validation',microbatches=4,pass_id=1)
        assert e.microbatches==4 and e.active and e.capture.pass_id==1
    e.active=False
    e.begin_iteration(0,phase='test',microbatches=4,pass_id=2)
    assert not e.active
    e.capture.set_enabled.assert_called_with(False)


def test_native_empty_record():
    from megatron.baseline_capture import Capture
    coords=dict(tp_rank=0,tp_size=1,pp_rank=0,pp_size=1,dp_rank=0)
    with patch('megatron.baseline_sites.parallel_coordinates',return_value=coords):
        cap=Capture(Model(),selected=['hidden_states'],mode=os.environ.get('BASELINE_CAPTURE_MODE','immediate'))
    device='cuda' if os.environ.get('BASELINE_TEST_CUDA') else 'cpu'
    cap.begin_iteration(0);cap.microbatch=0
    cap.forward(torch.empty(0,4,device=device));cap.end_iteration()
    assert len(cap.records)==1 and cap.records[0]['bytes']==0
    assert cap.records[0]['tensor'].shape==(0,4)
    cap.close()
