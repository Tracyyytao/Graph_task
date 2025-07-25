import time
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from models import GNNModel
from utils import train_full_graph, evaluate_link_prediction
from datasets.load_data import load_dataset

def run_experiment(model_type, dataset_name, use_sampling):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running {model_type} on {dataset_name} with {'sampling' if use_sampling else 'full-graph'}...")

    data, in_channels = load_dataset(dataset_name)
    data = data.to(device)

    model = GNNModel(model_type, in_channels, 64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    start_time = time.time()
    for epoch in range(1, 51):
        if use_sampling:
            loader = NeighborLoader(data, num_neighbors=[10, 5], batch_size=1024,
                                    input_nodes=torch.arange(data.num_nodes), shuffle=True)
            model.train()
            total_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                batch = batch.to(device)
                z = model(batch.x, batch.edge_index)
                pos_edge_index = data.train_pos_edge_index

                neg_edge_index = negative_sampling(
                    edge_index=pos_edge_index, num_nodes=data.num_nodes,
                    num_neg_samples=pos_edge_index.size(1)
                )

                pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
                neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)

                loss = F.binary_cross_entropy_with_logits(
                    torch.cat([pos_score, neg_score]),
                    torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))]).to(device)
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            loss = total_loss
        else:
            loss = train_full_graph(model, data, optimizer)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")

    z = model(data.x, data.train_pos_edge_index)
    auc, ap = evaluate_link_prediction(z, data.test_pos_edge_index, data.test_neg_edge_index)
    runtime = time.time() - start_time

    print(f"AUC: {auc:.4f}, AP: {ap:.4f}, Time: {runtime:.2f}s\n")
    return auc, ap, runtime

