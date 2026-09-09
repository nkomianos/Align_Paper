"""Offline single-GPU pilot backend, with explicit additive edit and LoRA reset."""
import json
import os
from pathlib import Path
from .common import score_rows, write

REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'


class Backend:
    def __init__(self, snapshot, out, seed):
        if os.environ.get('RESEARCH_PILOT_SUPERVISED') != '1':
            raise RuntimeError('use the bounded pilot supervisor')
        import torch
        from torch import nn
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM
        from latent_contract.sender_update import LoRALinear
        self.torch, self.out, self.counter = torch, out, 0
        snapshot = Path(snapshot)
        if snapshot.name != REVISION:
            raise ValueError('pinned Qwen3.5-9B snapshot required')
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_properties(0).total_memory < 70*1024**3:
            raise ValueError('need one GPU with >=70GiB memory; workload fit requires qualification')
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = Qwen3_5ForCausalLM.from_pretrained(snapshot, local_files_only=True,
            dtype=torch.bfloat16, device_map={'':0}, attn_implementation='sdpa', use_kernels=False).eval()
        self.model.requires_grad_(False)
        self.model.config.use_cache = False
        candidates = [(n,m) for n,m in self.model.named_modules()
                      if n.endswith('mlp.down_proj') and isinstance(m,nn.Linear)]
        if not candidates:
            raise ValueError('no supported MLP sites')
        self.site_name, base = candidates[len(candidates)//2]
        self.base_weight = base.weight.detach().clone()
        self.site = LoRALinear(base, rank=8, alpha=16)
        parent, _, leaf = self.site_name.rpartition('.')
        setattr(self.model.get_submodule(parent), leaf, self.site)
        self.initial = {n:p.detach().clone() for n,p in self.site.named_parameters() if p.requires_grad}
        ids = [self.tokenizer.encode(s,add_special_tokens=False) for s in ('A','B')]
        if any(len(x)!=1 for x in ids) or ids[0]==ids[1]:
            raise ValueError('answer token qualification failed')
        self.ids = [x[0] for x in ids]
        self.direction = None
        self.alpha = 0.
        self.graft = None
        self.last = None
        def hook(module, inputs, output):
            self.last = output[:, -1, :].detach().float()
            if self.graft is None:
                return output
            d, reference = self.graft
            replacement = output.clone()
            h = output[:, -1, :].float()
            replacement[:, -1, :] = (h + (reference - h@d)[:,None]*d).to(output.dtype)
            return replacement
        self.handle = self.site.register_forward_hook(hook)
        from interaction_sprint.hindsight_execution_integrity import capture_model_provenance
        write(out/'MODEL.json',capture_model_provenance(snapshot,'Qwen/Qwen3.5-9B',REVISION))
        write(out/'BACKEND.json',{'site':self.site_name,'answer_ids':self.ids,'seed':seed,
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'revision':REVISION})
        torch.save({'base_site_weight':self.base_weight.cpu(),'initial':{n:p.cpu() for n,p in self.initial.items()}},out/'INITIAL.pt')

    def reset(self):
        with self.torch.no_grad():
            for n,p in self.site.named_parameters():
                if n in self.initial:
                    p.copy_(self.initial[n])

    def set_edit(self, direction, alpha):
        # Reconstruct from original each time: no repeated add/subtract rounding drift.
        self.direction, self.alpha = direction, alpha
        with self.torch.no_grad():
            w = self.base_weight.float()
            if direction is not None and alpha:
                w = w + alpha * direction[:,None] * (direction@w)[None,:]
            self.site.base.weight.copy_(w.to(self.base_weight.dtype))

    def forward(self, row, prefix='', grad=False):
        t = self.torch
        rendered = self.tokenizer.apply_chat_template([{'role':'user','content':prefix+row['prompt']}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        enc = self.tokenizer(rendered,return_tensors='pt',add_special_tokens=False).to('cuda')
        if enc['input_ids'].shape[1] > 4096:
            raise ValueError('input over budget; truncation prohibited')
        with t.set_grad_enabled(grad):
            logits = self.model(**enc,use_cache=False,logits_to_keep=1).logits[0,-1].float()
        if not t.isfinite(logits).all():
            raise ValueError('nonfinite output')
        self.counter += 1
        record = {'id':row['id'],'target':row['target'],'choice_logits':logits[self.ids].detach().cpu().tolist(),
                  'logsumexp':float(logits.detach().logsumexp(-1)), 'prefix':prefix,
                  'alpha':self.alpha,'gradient':grad,'graft_active':self.graft is not None,
                  'prompt':row['prompt'],'rendered':rendered,
                  'input_ids':enc['input_ids'][0].cpu().tolist()}
        write(self.out/f'forward_{self.counter:06d}.json',record)
        record['forward_file'] = f'forward_{self.counter:06d}.json'
        return logits, record

    def score(self, rows, name, prefix=''):
        records = [self.forward(r,prefix)[1] for r in rows]
        value = {'rows':records,'summary':score_rows(records)}
        write(self.out/(name+'.json'),value)
        return value['summary']

    def capture(self, rows, prefix=''):
        values = []
        for r in rows:
            self.forward(r,prefix)
            values.append(self.last[0].clone())
        return self.torch.stack(values)

    def train(self, rows, name, steps=16):
        t = self.torch
        parameters = [p for p in self.site.parameters() if p.requires_grad]
        optimizer = t.optim.AdamW(parameters,lr=1e-4,weight_decay=0)
        # Eval mode is deliberate: no dropout; gradients still enabled.
        schedule = []
        for step in range(steps):
            batch = [rows[(step*4+i)%len(rows)] for i in range(4)]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for row in batch:
                logits,_ = self.forward(row,grad=True)
                loss = -logits.log_softmax(-1)[self.ids[row['target']]] / len(batch)
                loss.backward()
                losses.append(float(loss.detach()))
            norm = t.nn.utils.clip_grad_norm_(parameters,1.)
            if not t.isfinite(norm):
                raise ValueError('nonfinite gradient')
            optimizer.step()
            schedule.append({'step':step,'ids':[r['id'] for r in batch],'loss':sum(losses)})
        write(self.out/(name+'_schedule.json'),schedule)
        self.save(name)

    def save(self,name):
        self.torch.save({n:p.detach().cpu() for n,p in self.site.named_parameters() if p.requires_grad},self.out/(name+'.pt'))

    def load(self,name):
        state = self.torch.load(self.out/(name+'.pt'),weights_only=True,map_location='cuda')
        with self.torch.no_grad():
            for n,p in self.site.named_parameters():
                if p.requires_grad:
                    p.copy_(state[n])
