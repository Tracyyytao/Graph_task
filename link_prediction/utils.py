import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_link_prediction(z, pos_edge_index, neg_edge_index):
    with torch.no_grad():
        pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
        neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)

        scores = torch.cat([pos_score, neg_score]).cpu()
        labels = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))]).cpu()

        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        return auc, ap


def train_full_graph(model, data, optimizer):
    model.train()
    optimizer.zero_grad()

    # 使用训练边
    edge_index = data.train_pos_edge_index
    z = model(data.x, edge_index)

    # 负采样
    neg_edge_index = negative_sampling(
        edge_index=edge_index, num_nodes=data.num_nodes,
        num_neg_samples=edge_index.size(1)
    )

    pos_score = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
    neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)

    loss = F.binary_cross_entropy_with_logits(
        torch.cat([pos_score, neg_score]),
        torch.cat([torch.ones(pos_score.size(0)),
                   torch.zeros(neg_score.size(0))]).to(data.x.device)
    )
    loss.backward()
    optimizer.step()
    return loss.item()
