"""Fixed A0 in-microbatch triplet validation and detached audit statistics."""
import torch
from torch.nn import functional as F


def select_triplets(embedding, info, mode, margin):
    """Prefer same-level negatives, exclude same content and verified same groups.

    No cross-microbatch queue: gradient accumulation cannot enlarge this pool.
    Returns all anchor/positive indices, chosen negatives and an explicit mask.
    Missing candidates skip the scene term, never become same-scene negatives.
    """
    contents,levels = info['content_ids'],info['raw_levels']
    groups = info.get('group_ids',['']*len(contents))
    count = len(contents)
    if count != len(embedding) or len(levels) != count or len(groups) != count:
        raise ValueError('Triplet metadata length must match the physical microbatch.')
    if mode != 'fixed' or info['layout'] != 'triplet':
        raise ValueError('A0 supports only fixed triplet batches.')
    if count % 3: raise ValueError('Triplet layout must have N=3B.')
    anchors=list(range(count//3)); positives=[i+count//3 for i in anchors]
    for i,p in zip(anchors,positives):
        if i==p or contents[i]!=contents[p] or levels[i]==levels[p]:
            raise ValueError('Positive must be same content, different illumination, aligned spatially.')
    different=[[contents[j]!=contents[i] and not(groups[i] and groups[i]==groups[j])
                for j in range(count)] for i in anchors]
    same_level=[[legal and levels[i]==levels[j] for j,legal in enumerate(row)]
                for i,row in zip(anchors,different)]
    device=embedding.device
    legal=torch.tensor(different,device=device,dtype=torch.bool)
    preferred=torch.tensor(same_level,device=device,dtype=torch.bool)
    mask=torch.where(preferred.any(1,keepdim=True),preferred,legal)
    a=torch.tensor(anchors,device=device); p=torch.tensor(positives,device=device)
    detached=embedding.detach().float()
    dap=F.pairwise_distance(detached[a],detached[p],p=2,eps=1e-6)
    distances=F.pairwise_distance(detached[a,None,:],detached[None,:,:],p=2,eps=1e-6)
    valid=mask.any(1)
    n=a+2*(count//3)
    valid=mask.gather(1,n[:,None]).flatten()
    dan=distances.gather(1,n[:,None]).flatten()
    counts=mask.sum(1).float()
    stats=dict(triplet_anchor_count=valid.new_tensor(len(a),dtype=torch.float32),
        triplet_valid_count=valid.float().sum(),triplet_skipped=(~valid).float().sum(),
        valid_negative_count=counts.mean(),single_candidate_fraction=(counts==1).float().mean(),
        cross_level_fallback=(valid & ~preferred.any(1)).float().sum(),
        semihard_selected=dap.new_zeros(()),
        hard_fallback=dap.new_zeros(()),
        embedding_dimension_variance=detached.var(dim=0,unbiased=False).mean())
    if valid.any():
        da,dn=dap[valid],dan[valid]
        stats.update(d_ap=da.mean(),d_an=dn.mean(),active_triplet_ratio=(da-dn+margin>0).float().mean())
        for name,value in (('d_ap',da),('d_an',dn)):
            for suffix,q in (('p10',.1),('median',.5),('p90',.9)):
                stats[name+'_'+suffix]=torch.quantile(value,q)
    return a,p,n,valid,stats
