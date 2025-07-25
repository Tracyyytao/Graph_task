from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import train_test_split_edges


def load_dataset(name):
    if name in ['Cora', 'Citeseer']:
        dataset = Planetoid(root=f'data/{name}', name=name, transform=ToUndirected())
    elif name == 'Flickr':
        dataset = Flickr(root='data/Flickr', transform=ToUndirected())
    else:
        raise ValueError("Unsupported dataset")

    data = dataset[0]
    data = train_test_split_edges(data)
    return data, dataset.num_node_features
