import torch


# From DisCo github: https://github.com/gkasieczka/DisCo/blob/efdcda00ecdafffba61dd86b9404c6100eeb56f1/Disco.py
# Paper: https://arxiv.org/abs/2001.05310
def distance_corr(var_1,var_2,normedweight,power=1):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation
    
    var1_1, var_2 and normedweight should all be 1D torch tensors with the same number of entries
    
    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """
    xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))
    yy = var_1.repeat(len(var_1),1).view(len(var_1),len(var_1))
    amat = (xx-yy).abs()

    xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))
    yy = var_2.repeat(len(var_2),1).view(len(var_2),len(var_2))
    bmat = (xx-yy).abs()

    amatavg = torch.mean(amat * normedweight,dim=1)
    Amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))\
        -amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))\
        +torch.mean(amatavg * normedweight)

    bmatavg = torch.mean(bmat * normedweight,dim=1)
    Bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
        -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
        +torch.mean(bmatavg * normedweight)

    ABavg = torch.mean(Amat * Bmat * normedweight,dim=1)
    AAavg = torch.mean(Amat * Amat * normedweight,dim=1)
    BBavg = torch.mean(Bmat * Bmat * normedweight,dim=1)

    if power == 1:
        dCorr = (torch.mean(ABavg * normedweight)) / torch.sqrt((torch.mean(AAavg * normedweight) * torch.mean(BBavg * normedweight)))
    elif power == 2:
        dCorr=(torch.mean(ABavg * normedweight))**2/(torch.mean(AAavg * normedweight) * torch.mean(BBavg * normedweight))
    else:
        dCorr=((torch.mean(ABavg * normedweight))/torch.sqrt((torch.mean(AAavg * normedweight) * torch.mean(BBavg * normedweight))))**power

    return dCorr



def distance_corr_multi(var_1, var_2_2d, normedweight, power=1, reduce='mean', class_indices=None):
    """
    Vectorized distance correlation between var_1 (N,) and each column of var_2_2d (N,C).

    Args:
        var_1 (torch.Tensor): 1D tensor of shape (N,) to decorrelate, e.g. mass.
        var_2_2d (torch.Tensor): 2D tensor of shape (N,C) to decorrelate against, e.g. classifier outputs.
        normedweight (torch.Tensor): weights normalized so sum equals N.
        power (int): Exponent used in calculating the distance correlation.
        reduce (str): 'mean'|'sum'|'max'|'quadrature'|'none' to aggregate across classes.
        class_indices (list[int] | None): optional list/1D tensor selecting columns from var_2_2d.
    """
    var_1 = var_1.reshape(-1)
    if var_2_2d.dim() == 1:
        return distance_corr(var_1, var_2_2d.reshape(-1), normedweight, power=power)

    if class_indices is not None:
        var_2_2d = var_2_2d[:, class_indices]

    N, C = var_2_2d.shape

    # Pairwise distances for var_1 (shared)
    x = var_1
    xx = x.view(N, 1).repeat(1, N)
    yy = x.view(1, N).repeat(N, 1)
    amat = (xx - yy).abs()

    # Pairwise distances per class
    y = var_2_2d
    bmat = (y.unsqueeze(1) - y.unsqueeze(0)).abs()  # [N,N,C]

    w = normedweight.reshape(N)
    w_col = w.view(1, N, 1)
    w_row = w.view(N, 1, 1)

    # Center A
    amatavg = (amat * w.view(1, N)).mean(dim=1)  # [N]
    Amat = amat - amatavg.view(1, N) - amatavg.view(N, 1) + (amatavg * w).mean()  # [N,N]

    # Center B per class
    bmatavg = (bmat * w_col).mean(dim=1)  # [N,C]
    Bmat = bmat - bmatavg.unsqueeze(1) - bmatavg.unsqueeze(0) + (bmatavg * w.view(N,1)).mean(dim=0).view(1,1,C)  # [N,N,C]

    # AB, AA, BB
    ABavg = (Amat.unsqueeze(-1) * Bmat * w_col).mean(dim=1)  # [N,C]
    AAavg = ((Amat * Amat) * w.view(1, N)).mean(dim=1)       # [N]
    BBavg = (Bmat * Bmat * w_col).mean(dim=1)                # [N,C]

    num = (ABavg * w.view(N,1)).mean(dim=0)                  # [C]
    den = torch.sqrt((AAavg * w).mean() * (BBavg * w.view(N,1)).mean(dim=0) + 1e-12)  # [C]
    dCorr = num / (den + 1e-12)                              # [C]

    if power == 2:
        dCorr = dCorr * dCorr
    elif power != 1:
        dCorr = torch.pow(dCorr, power)

    if reduce == 'mean':
        return dCorr.mean()
    elif reduce == 'sum':
        return dCorr.sum()
    elif reduce == 'max':
        return dCorr.max()
    elif reduce == 'quadrature':
        return torch.sqrt((dCorr * dCorr).sum())
    else:
        return dCorr