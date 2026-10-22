"""A0–A2: Stage 1 models; A3: fairness-aware masked GCN.
Datasets: Bail, Credit, German, with original baseline graph construction and splits.
"""
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, f1_score


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_npz(path, device):
    arr=np.load(path,allow_pickle=False)
    required=['x','edge_index','y','sensitive','train_idx','val_idx','test_idx']
    missing=[k for k in required if k not in arr]
    if missing: raise ValueError(f'{path}: missing {missing}')
    x=torch.as_tensor(arr['x'],dtype=torch.float32,device=device)
    ei=torch.as_tensor(arr['edge_index'],dtype=torch.long,device=device)
    y=torch.as_tensor(arr['y'],dtype=torch.long,device=device).view(-1)
    s=torch.as_tensor(arr['sensitive'],dtype=torch.long,device=device).view(-1)
    if ei.shape[0]!=2 and ei.shape[1]==2: ei=ei.T.contiguous()
    assert ei.shape[0]==2 and ei.numel()>0
    n=len(x)
    if x.ndim != 2 or not torch.isfinite(x).all(): raise ValueError('x must be a finite [N,F] matrix')
    if len(y)!=n or len(s)!=n: raise ValueError('labels/sensitive must match node count')
    if set(y.unique().tolist()) != {0, 1}: raise ValueError('both task labels must be present')
    if ei.min()<0 or ei.max()>=n: raise ValueError('edge indices out of range')
    if set(y.unique().tolist())-set([0,1]): raise ValueError('binary y required')
    if set(s.unique().tolist())-set([0,1]): raise ValueError('binary sensitive required')
    idx={k:torch.as_tensor(arr[k],dtype=torch.long,device=device).view(-1) for k in ['train_idx','val_idx','test_idx']}
    for k,v in idx.items():
        if v.numel()==0 or v.min()<0 or v.max()>=n or len(v.unique())!=len(v): raise ValueError('bad '+k)
    sets=[set(v.cpu().tolist()) for v in idx.values()]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)): raise ValueError('splits overlap')
    return x,ei,y,s,idx


def undirected_edges(edge_index, num_nodes):
    a,b=edge_index
    keep=a!=b
    a,b=a[keep],b[keep]
    lo,hi=torch.minimum(a,b),torch.maximum(a,b)
    pairs=torch.unique(torch.stack((lo,hi),dim=1),dim=0)
    return pairs.T.contiguous()


class GCNLayer(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__(); self.lin=nn.Linear(in_dim,out_dim)
    def forward(self,x,edge_index,weights):
        n=x.size(0); u,v=edge_index
        # Symmetric weighted GCN normalization; graph edges supplied once as undirected pairs.
        u2=torch.cat([u,v,torch.arange(n,device=x.device)])
        v2=torch.cat([v,u,torch.arange(n,device=x.device)])
        w=torch.cat([weights,weights,torch.ones(n,device=x.device)])
        deg=torch.zeros(n,device=x.device).index_add(0,v2,w)
        norm=w*(deg[u2].clamp_min(1e-8)*deg[v2].clamp_min(1e-8)).rsqrt()
        h=self.lin(x)
        out=torch.zeros_like(h).index_add(0,v2,h[u2]*norm[:,None])
        return out


class MaskGCN(nn.Module):
    def __init__(self,nfeatures,hidden,dropout=0.3,masked=False):
        super().__init__()
        self.masked=masked; self.dropout=dropout
        self.pre=nn.Linear(nfeatures,hidden)
        self.g1=GCNLayer(nfeatures,hidden)
        self.g2=GCNLayer(hidden,2)
        if masked:
            self.mask_mlp=nn.Sequential(nn.Linear(hidden*3,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def get_mask(self,x,ei):
        if not self.masked:return torch.ones(ei.size(1),device=x.device)
        h=torch.relu(self.pre(x)); u,v=ei
        z=torch.cat([h[u]+h[v],(h[u]-h[v]).abs(),h[u]*h[v]],dim=1)
        return torch.sigmoid(self.mask_mlp(z).flatten())
    def forward(self,x,ei,force_full=False):
        m=self.get_mask(x,ei)
        w=torch.ones_like(m) if force_full else m
        h=F.relu(self.g1(x,ei,w));h=F.dropout(h,p=self.dropout,training=self.training)
        return self.g2(h,ei,w),m


def safe_auc(y,score):
    return float(roc_auc_score(y,score)) if len(np.unique(y))==2 else float('nan')


def metrics(logits,y,s,indices):
    pr=logits.softmax(-1)[:,1].detach().cpu().numpy()
    yt=y.detach().cpu().numpy(); st=s.detach().cpu().numpy()
    ids=indices.detach().cpu().numpy(); a=yt[ids]; b=(pr[ids]>=0.5).astype(int); sen=st[ids]
    rates=[]; tprs=[]
    for g in [0,1]:
        mask=sen==g
        rates.append(float(b[mask].mean()) if mask.any() else float('nan'))
        positive=mask&(a==1)
        tprs.append(float(b[positive].mean()) if positive.any() else float('nan'))
    return {'f1':float(f1_score(a,b,zero_division=0)),'auc':safe_auc(a,pr[ids]),
            'parity':abs(rates[0]-rates[1]),'equality':abs(tprs[0]-tprs[1]),
            'positive_prediction_rate':float(b.mean())}


def differentiable_fairness(logits, y, sensitive, train_idx):
    """Train-only soft DP and EO differences (binary positive-class probabilities)."""
    probabilities = logits.softmax(-1)[:, 1]
    labels, groups = y[train_idx], sensitive[train_idx]
    probs = probabilities[train_idx]
    g0, g1 = groups == 0, groups == 1
    if not (g0.any() and g1.any() and (g0 & (labels == 1)).any() and (g1 & (labels == 1)).any()):
        raise ValueError('A3 requires both sensitive groups and positive labels within both groups in TRAIN')
    dp = (probs[g0].mean() - probs[g1].mean()).square()
    eo = (probs[g0 & (labels == 1)].mean() - probs[g1 & (labels == 1)].mean()).square()
    return dp, eo


def mask_exposure_gap(mask, edge_index, sensitive, train_idx, num_nodes):
    """Degree-normalized incident edge importance gap on selected nodes.

    A group-level descriptive proxy, not a causal sensitive-direction projection.
    """
    u, v = edge_index
    sums = torch.zeros(num_nodes, device=mask.device).index_add(0, u, mask).index_add(0, v, mask)
    counts = torch.zeros(num_nodes, device=mask.device).index_add(0, u, torch.ones_like(mask)).index_add(0, v, torch.ones_like(mask))
    has_edges = counts > 0
    group0 = (sensitive == 0) & has_edges
    group1 = (sensitive == 1) & has_edges
    selected = torch.zeros(num_nodes, dtype=torch.bool, device=mask.device)
    selected[train_idx] = True
    group0, group1 = group0 & selected, group1 & selected
    if not group0.any() or not group1.any():
        raise ValueError('Need nodes with incident edges from both sensitive groups')
    exposure = sums / counts.clamp_min(1)
    return (exposure[group0].mean() - exposure[group1].mean()).abs()


def train_one(data,model_name,seed,args,device):
    set_seed(seed)
    x,ei_raw,y,s,idx=data
    ei=undirected_edges(ei_raw,len(x))
    model=MaskGCN(x.size(1),args.hidden,args.dropout,model_name!='A0').to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    # A2/A3 use a frozen A0 reference trained on the same seed and split.
    teacher=None
    if model_name in ('A2', 'A3'):
        teacher=MaskGCN(x.size(1),args.hidden,args.dropout,False).to(device)
        ckpt=Path(args.output)/f'{args.dataset}_A0_seed{seed}.pt'
        if not ckpt.exists(): raise FileNotFoundError(f'Run A0 first; teacher missing: {ckpt}')
        teacher.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True))
        teacher.eval();teacher.requires_grad_(False)
        with torch.no_grad(): reference=teacher(x,ei)[0].softmax(-1)
    best=-float('inf');best_state=None;bad=0
    for epoch in range(args.epochs):
        model.train();opt.zero_grad()
        logits,m=model(x,ei)
        loss=F.cross_entropy(logits[idx['train_idx']],y[idx['train_idx']])
        if model_name in ('A2', 'A3'):
            # Distillation/reference fidelity from frozen full-graph A0 teacher, on TRAIN nodes only.
            pred=F.log_softmax(logits[idx['train_idx']],dim=-1)
            fidelity=F.kl_div(pred,reference[idx['train_idx']],reduction='batchmean')
            loss=loss+args.lambda_fid*fidelity+args.lambda_sparse*m.mean()
        if model_name == 'A3':
            dp, eo = differentiable_fairness(logits, y, s, idx['train_idx'])
            exposure = mask_exposure_gap(m, ei, s, idx['train_idx'], len(x))
            loss = loss + args.lambda_dp*dp + args.lambda_eo*eo + args.lambda_mask_fair*exposure.square()
        loss.backward();opt.step()
        model.eval()
        with torch.no_grad(): vl,_=model(x,ei);vmet=metrics(vl,y,s,idx['val_idx'])
        score=vmet['auc'] if np.isfinite(vmet['auc']) else vmet['f1']
        if not np.isfinite(score): raise ValueError('Validation score undefined; inspect labels and split')
        if score>best+1e-7:
            best=score;bad=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=args.patience:break
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():lg,mask=model(x,ei);out=metrics(lg,y,s,idx['test_idx'])
    if model_name != 'A0':
        with torch.no_grad():
            out['mask_exposure_gap'] = float(mask_exposure_gap(mask,ei,s,idx['test_idx'],len(x)))
    else:
        out['mask_exposure_gap'] = float('nan')
    out.update({'dataset':args.dataset,'model':model_name,'seed':seed,
        'mask_mean':float(mask.mean()),'mask_frac_gt_05':float((mask>0.5).float().mean()),
        'best_validation_score':float(best),'epochs_trained':epoch+1})
    torch.save(model.state_dict(),Path(args.output)/f'{args.dataset}_{model_name}_seed{seed}.pt')
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',required=True);p.add_argument('--dataset',choices=['bail','credit','german'],required=True)
    p.add_argument('--output',default='results');p.add_argument('--models',nargs='+',choices=['A0','A1','A2','A3'],default=['A0','A1','A2','A3'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    p.add_argument('--epochs',type=int,default=300);p.add_argument('--patience',type=int,default=40)
    p.add_argument('--hidden',type=int,default=64);p.add_argument('--dropout',type=float,default=.3)
    p.add_argument('--lr',type=float,default=.005);p.add_argument('--weight-decay',type=float,default=0.0005)
    p.add_argument('--lambda-dp',type=float,default=1.0);p.add_argument('--lambda-eo',type=float,default=1.0)
    p.add_argument('--lambda-mask-fair',type=float,default=0.1)
    p.add_argument('--lambda-fid',type=float,default=1.0);p.add_argument('--lambda-sparse',type=float,default=.005)
    args=p.parse_args();Path(args.output).mkdir(parents=True,exist_ok=True)
    result_file=Path(args.output)/'stage2_results.csv'
    if result_file.exists(): raise FileExistsError(f'{result_file} already exists; choose a new output directory')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if min(args.lambda_fid,args.lambda_sparse,args.lambda_dp,args.lambda_eo,args.lambda_mask_fair) < 0: raise ValueError('Loss weights must be nonnegative')
    for seed in args.seeds:
        # Optional split-per-seed NPZ takes priority; otherwise fixed split for all seeds.
        specific=Path(args.data_dir)/f'{args.dataset}_seed{seed}.npz'
        path=specific if specific.exists() else Path(args.data_dir)/f'{args.dataset}.npz'
        if not path.exists():raise FileNotFoundError(f'Missing {path}; supply raw graph data and original splits')
        data=load_npz(path,device)
        for name in ['A0','A1','A2','A3']:
            if name not in args.models:continue
            row=train_one(data,name,seed,args,device)
            outfile=Path(args.output)/'stage2_results.csv'; exists=outfile.exists()
            with outfile.open('a',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row));
                if not exists:w.writeheader()
                w.writerow(row)
            print(json.dumps(row,allow_nan=True))

if __name__=='__main__':main()
