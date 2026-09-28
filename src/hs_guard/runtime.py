# SPDX-License-Identifier: Apache-2.0
"""Load the released classifier and score complete messages or stable token streams."""
from pathlib import Path
import hashlib
import json

LABELS = ('safe', 'unsafe', 'controversial')
MODEL_ID = 'KrisLiu16/HS-Guard-v10'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_model(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / 'hs_guard.json').read_text())
    if manifest.get('format') != 'hs-guard-v1' or manifest.get('risk_labels') != list(LABELS):
        raise ValueError('Unsupported model format or risk-label order')
    required = {'config.json', 'tokenizer.json', 'model.safetensors'}
    if not required.issubset(manifest.get('files', {})):
        raise ValueError('Model manifest is missing required assets')
    for name, expected in manifest['files'].items():
        p = root / name
        if p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file():
            raise ValueError(f'Invalid model asset: {name}')
        if sha256(p) != expected:
            raise ValueError(f'Model asset checksum mismatch: {name}')
    return manifest


def serialize(messages):
    if not messages or messages[-1]['role'] not in ('user', 'assistant'):
        raise ValueError('The last message must have role user or assistant')
    if any(m['role'] not in ('system', 'user', 'assistant') or not isinstance(m['content'], str) for m in messages):
        raise ValueError('Messages require a supported role and string content')
    if not messages[-1]['content']:
        raise ValueError('The target message is empty')
    return '\n\n'.join(m['role'].upper() + ':\n' + m['content'] for m in messages)


def _build(backbone):
    import torch
    from torch import nn
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            def head(categories):
                return nn.ModuleDict({
                    'projection': nn.Sequential(nn.Linear(backbone.config.hidden_size, 512), nn.LayerNorm(512), nn.SiLU()),
                    'risk': nn.Linear(512, 3), 'category': nn.Linear(512, categories),
                    'general_projection': nn.Sequential(nn.Linear(backbone.config.hidden_size, 512), nn.LayerNorm(512), nn.SiLU()),
                    'general': nn.Linear(512, 3), 'general_category': nn.Linear(512, categories)})
            self.heads = nn.ModuleDict({'user': head(9), 'assistant': head(8)})
        def forward(self, input_ids, attention_mask=None, **kwargs):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        def readout(self, hidden, role):
            head = self.heads[role]
            value = head['projection'](hidden.float())
            return head['risk'](value), head['category'](value)
    return Classifier()


def _enable_fla():
    import transformers.models.qwen3_5.modeling_qwen3_5 as module
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    def wrap(fn):
        def call(q,k,v,g,beta,initial_state=None,output_final_state=False,
                 use_qk_l2norm_in_kernel=True,cu_seqlens=None,**kwargs):
            return fn(q,k,v,g=g,beta=beta,initial_state=initial_state,
                      output_final_state=output_final_state,
                      use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,cu_seqlens=cu_seqlens)
        return call
    module.torch_chunk_gated_delta_rule=wrap(chunk_gated_delta_rule)
    module.torch_recurrent_gated_delta_rule=wrap(fused_recurrent_gated_delta_rule)


def load_model(model=MODEL_ID, *, revision=None, head='redline'):
    if head not in ('redline', 'general'):
        raise ValueError('head must be redline or general')
    root = Path(model)
    if not root.is_dir():
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(str(model), revision=revision))
    metadata = verify_model(root)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('HS-Guard requires an NVIDIA CUDA GPU')
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    from tokenizers import Tokenizer
    from safetensors.torch import load_file
    from .window_attention import set_window
    config = Qwen3_5TextConfig.from_dict(json.loads((root / 'config.json').read_text()))
    config._attn_implementation = 'sdpa'
    _enable_fla()
    # RoPE buffers are not checkpoint tensors and must retain constructor precision.
    with torch.device('cpu'):
        backbone = Qwen3_5TextModel(config)
    with torch.no_grad():
        for p in backbone.parameters():
            p.data = p.data.to(torch.bfloat16)
    set_window(backbone, metadata['window_tokens'])
    classifier = _build(backbone).cuda()
    classifier.load_state_dict(load_file(str(root / 'model.safetensors')), strict=True)
    if head == 'general':
        for h in classifier.heads.values():
            h['projection'], h['risk'], h['category'] = h['general_projection'], h['general'], h['general_category']
    classifier.eval().requires_grad_(False)
    metadata = dict(metadata, head=head, thresholds=metadata['thresholds'][head])
    return classifier, Tokenizer.from_file(str(root / 'tokenizer.json')), metadata


def encode_messages(tokenizer, messages):
    text = serialize(messages)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    start = len(text) - len(messages[-1]['content'])
    positions = [i for i, (_, end) in enumerate(encoded.offsets) if end > start]
    return encoded.ids, positions, messages[-1]['role']


def score_messages(model, tokenizer, metadata, messages):
    import torch
    ids, positions, role = encode_messages(tokenizer, messages)
    with torch.inference_mode():
        x = torch.tensor([ids], device='cuda')
        hidden = model(x, attention_mask=torch.ones_like(x), use_cache=False).last_hidden_state[0]
        cut = (1 - model.readout(hidden[positions], role)[0].float().softmax(-1)[:, 0]).cpu()
    value = float(cut[-1] if role == 'user' else cut.max())
    threshold = metadata['thresholds'][role]
    return {'role': role, 'head': metadata['head'], 'score': value, 'threshold': threshold, 'flagged': value > threshold}


class StreamingGuard:
    """Callers provide stable token IDs and keep each live conversation in its own slot."""
    def __init__(self, model, metadata, max_slots=1):
        if not isinstance(max_slots, int) or max_slots < 1:
            raise ValueError('max_slots must be a positive integer')
        from .engine import SlotStreamEngineV8
        buckets = tuple(x for x in (1,2,4,8,16,32,48,64,96,128,160,192,224,256,320,352,384,448,512) if x <= max_slots)
        if not buckets or buckets[-1] != max_slots:
            buckets += (max_slots,)
        self.engine = SlotStreamEngineV8(model,max_slots,state_dtype='float16',ring='inplace',gdn_bv=32,gdn_warps=2,session_buckets=buckets)
        self.metadata, self.vocab_size, self.max_slots = metadata, model.backbone.config.vocab_size, max_slots
    def reset(self):
        self.engine.reset()
    def step(self, requests, role='assistant'):
        if role not in ('user', 'assistant'):
            raise ValueError('role must be user or assistant')
        slots = [s for s,_ in requests]
        if not requests or len(set(slots)) != len(slots):
            raise ValueError('Each tick requires nonempty requests and unique slots')
        for slot, ids in requests:
            if not isinstance(slot,int) or not 0 <= slot < self.max_slots:
                raise ValueError('Slot is outside the allocated range')
            if not ids or any(not isinstance(i,int) or not 0 <= i < self.vocab_size for i in ids):
                raise ValueError('Each request requires nonempty in-vocabulary integer token IDs')
        outputs = self.engine.step(requests)[role]
        tau = self.metadata['thresholds'][role]
        return [{'slot':s,'cut_scores':(1-p[:,0]).tolist(),'threshold':tau} for (s,_),p in zip(requests,outputs)]
