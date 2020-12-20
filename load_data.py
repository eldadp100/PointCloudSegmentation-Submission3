from torch_geometric import datasets
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader


def get_data(batch_size=32):
    # load datasets
    train_dataset = datasets.ShapeNet(root='./data/shape_net', split='train', pre_transform=T.KNNGraph(k=20),
                                      transform=T.RandomTranslate(0.01))
    test_dataset = datasets.ShapeNet(root='./data/shape_net', split='val', pre_transform=T.KNNGraph(k=20),
                                     transform=T.RandomTranslate(0.01))

    # create dataloaders
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dl, test_dl

