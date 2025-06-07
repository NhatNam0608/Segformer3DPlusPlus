import torch
from typing import Tuple, Callable

def do_nothing(x: torch.Tensor, mode: str = None):
    return x

def bipartite_soft_matching_random3d(metric: torch.Tensor,
                                     d: int, h: int, w: int,
                                     sx: int, sy: int, sz: int,
                                     r: int, rand: bool = False) -> Tuple[Callable, Callable]:
    """
    3D version of bipartite soft matching with random partitioning.
    Args:
        metric [B, N, C]: similarity metric
        d, h, w: depth, height, width of the image in tokens
        sx, sy, sz: strides along width, height, and depth (must divide w, h, d)
        r: number of tokens to merge (from src to dst)
    Returns:
        merge and unmerge functions
    """
    B, N, _ = metric.shape
    if r <= 0:
        return do_nothing, do_nothing

    gather = torch.gather

    with torch.no_grad():
        dsz, hsy, wsx = d // sz, h // sy, w // sx  # output grid size
        if rand:
            rand_idx = torch.randint(0, sz * sy * sx, (dsz, hsy, wsx, 1), device=metric.device)
        else:
            rand_idx = torch.zeros(dsz, hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        
        idx_buffer_view = torch.zeros(dsz, hsy, wsx, sz * sy * sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=3, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=torch.int64))
        idx_buffer_view = idx_buffer_view.view(dsz, hsy, wsx, sz, sy, sx).permute(0, 3, 1, 4, 2, 5).reshape(dsz*sz, hsy*sy, wsx*sx)

        if (dsz * sz) < d or (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(d, h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(dsz * sz), :(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        num_dst = dsz * hsy * wsx
        a_idx = rand_idx[:, num_dst:, :]  # src
        b_idx = rand_idx[:, :num_dst, :]  # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        r_eff = min(a.shape[1], r)
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r_eff:, :]
        src_idx = edge_idx[..., :r_eff, :]
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape

        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r_eff, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r_eff, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r_eff, c), src, reduce=mode)
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r_eff, c))
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r_eff, c), src=src)
        return out

    return merge, unmerge