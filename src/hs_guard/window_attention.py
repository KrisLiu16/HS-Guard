# SPDX-License-Identifier: Apache-2.0
from types import MethodType
import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb


class WindowCache(DynamicCache):
    def __init__(self, config, window):
        super().__init__(config=config)
        self.window=window
        self.seen={}
        self.first_attention=config.layer_types.index('full_attention')

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        length=key_states.shape[-2]
        keys,values=super().update(key_states,value_states,layer_idx,*args,**kwargs)
        self.seen[layer_idx]=self.seen.get(layer_idx,0)+length
        layer=self.layers[layer_idx]
        keep=self.window-1
        # clone is deliberate: a small slice must not retain the full storage.
        layer.keys=keys[..., -keep:, :].clone() if keep else keys[..., :0, :].clone()
        layer.values=values[..., -keep:, :].clone() if keep else values[..., :0, :].clone()
        return keys,values

    def get_seq_length(self, layer_idx=0):
        return self.seen.get(layer_idx,self.seen.get(self.first_attention,0))

    def crop(self, *args, **kwargs):
        raise NotImplementedError('GDN rollback needs a saved state and replay; cropping KV alone is invalid')


def tiled_window_attention(q,k,v,window,query_block=128,key_mask=None,scale=None,dropout_p=0.):
    length=q.shape[-2];history=k.shape[-2]-length
    if history<0:raise ValueError('K/V cannot be shorter than the current query')
    groups=q.shape[1]//k.shape[1]
    pieces=[]
    for start in range(0,length,query_block):
        end=min(length,start+query_block)
        left=max(0,history+start-window+1);right=history+end
        local_k=k[...,left:right,:];local_v=v[...,left:right,:]
        if groups!=1:
            local_k=local_k.repeat_interleave(groups,dim=1)
            local_v=local_v.repeat_interleave(groups,dim=1)
        qi=torch.arange(history+start,history+end,device=q.device)[:,None]
        ki=torch.arange(left,right,device=q.device)[None,:]
        allowed=((ki<=qi)&(ki>qi-window))[None,None,:,:]
        if key_mask is not None:
            allowed=allowed & key_mask[:,None,None,left:right].bool()
        pieces.append(F.scaled_dot_product_attention(q[...,start:end,:],local_k,local_v,
                      attn_mask=allowed,dropout_p=dropout_p,is_causal=False,scale=scale))
    return torch.cat(pieces,dim=-2)


def attention_forward(self,hidden_states,position_embeddings,attention_mask=None,past_key_values=None,**kwargs):
    shape=hidden_states.shape[:-1];head_shape=(*shape,-1,self.head_dim)
    query,gate=torch.chunk(self.q_proj(hidden_states).view(*shape,-1,self.head_dim*2),2,dim=-1)
    gate=gate.reshape(*shape,-1)
    query=self.q_norm(query.reshape(head_shape)).transpose(1,2)
    key=self.k_norm(self.k_proj(hidden_states).view(head_shape)).transpose(1,2)
    value=self.v_proj(hidden_states).view(head_shape).transpose(1,2)
    query,key=apply_rotary_pos_emb(query,key,*position_embeddings)
    if past_key_values is not None:
        if not isinstance(past_key_values,WindowCache):raise TypeError('Window attention requires WindowCache')
        key,value=past_key_values.update(key,value,self.layer_idx)
    output=tiled_window_attention(query,key,value,self.guard_window,self.guard_query_block,
                                 attention_mask,self.scaling,self.attention_dropout if self.training else 0.)
    output=output.transpose(1,2).reshape(*shape,-1).contiguous()
    return self.o_proj(output*torch.sigmoid(gate)),None


def backbone_forward(self,input_ids=None,attention_mask=None,past_key_values=None,use_cache=False,**kwargs):
    if isinstance(attention_mask,dict):raise ValueError('Pass a 2D padding mask, not a prebuilt attention mask')
    if use_cache or past_key_values is not None:
        if self.training:raise ValueError('Cached training needs explicit truncated-BPTT semantics')
        if attention_mask is not None and not bool(torch.all(attention_mask==1)):
            raise ValueError('Padded cached batches are not supported in this research implementation')
        if past_key_values is None:past_key_values=WindowCache(self.config,self.guard_window)
        if not isinstance(past_key_values,WindowCache):raise TypeError('Cannot reuse full-attention cache after changing architecture')
        masks={'full_attention':None,'linear_attention':None}
    else:
        masks={'full_attention':attention_mask,'linear_attention':attention_mask}
    return self.guard_original_forward(input_ids=input_ids,attention_mask=masks,
             past_key_values=past_key_values,use_cache=use_cache,**kwargs)


def set_window(backbone,window,query_block=128):
    if window is not None and (window<2 or query_block<1):raise ValueError('Invalid window/tile size')
    if not hasattr(backbone,'guard_original_forward'):
        backbone.guard_original_forward=backbone.forward
        for layer in backbone.layers:
            if hasattr(layer,'self_attn'):
                layer.self_attn.guard_original_forward=layer.self_attn.forward
    backbone.guard_window=window
    backbone.forward=backbone.guard_original_forward if window is None else MethodType(backbone_forward,backbone)
    for layer in backbone.layers:
        if hasattr(layer,'self_attn'):
            attn=layer.self_attn;attn.guard_window=window;attn.guard_query_block=query_block
            attn.forward=attn.guard_original_forward if window is None else MethodType(attention_forward,attn)


def cache_storage(cache):
    storages={};kv=[]
    def visit(value):
        if isinstance(value,torch.Tensor):
            storage=value.untyped_storage();storages[(str(value.device),storage.data_ptr())]=storage.nbytes()
        elif isinstance(value,(tuple,list)):
            for item in value:visit(item)
        elif isinstance(value,dict):
            for item in value.values():visit(item)
    for index,layer in enumerate(cache.layers):
        visit(vars(layer))
        key=getattr(layer,'keys',None)
        if key is not None and key.ndim==4:
            kv.append({'layer':index,'retained_tokens':key.shape[-2],
                       'key_storage_bytes':key.untyped_storage().nbytes()})
    return {'unique_storage_bytes':sum(storages.values()),'attention_layers':kv,
            'logical_tokens_seen':cache.get_seq_length()}
