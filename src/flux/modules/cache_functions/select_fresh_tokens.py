import torch

def get_indices_by_random(cluster_info):
    '''
    选取 K * cluster_nums 个随机索引，相比于从每个聚类中各选 K 个实现更简单，速度更快
    '''
    cluster_indices, cluster_num, K = cluster_info['cluster_indices'], cluster_info['cluster_num'], cluster_info['topk']
    B, N = cluster_indices.shape
    device = cluster_indices.device

    fresh_indices = torch.randn((B, N), device=device).argsort(dim=1)[:, :K * cluster_num]

    return fresh_indices

def select_one_fresh_index_per_cluster(cache_dic, current):
    '''
    从每个聚类中恰好只选一个 fresh index
    '''
    cluster_info = cache_dic['cluster_info'][current['stream']][current['module']]
    cluster_indices, cluster_num, K = cluster_info['cluster_indices'], cluster_info['cluster_num'], cluster_info['topk']
    B, N = cluster_indices.shape

    device = cluster_indices.device
    rand_weights = torch.rand((B, N), device=device)

    cluster_ids = torch.arange(cluster_num, device=device).view(1, -1, 1) # [1, 32, 1]

    mask = (cluster_indices.unsqueeze(1) == cluster_ids) # [B, K, N]:[1, 32, 4096]

    masked_weights = torch.where(mask, rand_weights.unsqueeze(1), torch.tensor(-float('inf'), device=device)) # [B, K, N]:[1, 32, 4096]

    fresh_indices = masked_weights.argmax(dim=2, keepdim=False) # [B, K]:[1, 32]

    return fresh_indices

def select_fresh_indices_randomly(tokens, topk):
    '''
    随机选择topk个索引(用于和ToCa比较)
    '''
    B, N, D = tokens.shape
    device = tokens.device
    fresh_indices = torch.randn((B, N), device=device).argsort(dim=1)[:, :topk]
    return fresh_indices