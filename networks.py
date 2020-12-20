import torch
from torch import nn
from torch_geometric.nn import EdgeConv, EdgePooling
from torch_geometric.data import Data, Batch  # for test


# class GraphConvolution(nn.Module):
#     pass
#
# class GraphPooling(nn.Module):
#     pass


class UNetBlock(nn.Module):
    def __init__(self, input_features, sub_net_input_features, sub_network):
        super(UNetBlock, self).__init__()
        # define down_conv
        lin_in = nn.Linear(input_features * 2, sub_net_input_features)
        self.down_conv = EdgeConv(lin_in, aggr='mean')
        # pooling
        self.pooling = EdgePooling(sub_net_input_features)
        # define up_conv
        lin_out = nn.Linear(sub_net_input_features * 2, input_features)
        self.up_conv = EdgeConv(lin_out, aggr='mean')

        self.sub_network = sub_network

    def forward(self, nodes_feat, edges_index, batch):
        out = self.down_conv(nodes_feat, edges_index)
        if self.sub_network is not None:
            out, edges_index, batch, unpool_info = self.pooling(out, edges_index, batch)
            out = self.sub_network(out, edges_index, batch)
            out, edges_index, batch = self.pooling.unpool(out, unpool_info)
        out = self.up_conv(out, edges_index)

        # add residual connection
        out = out + nodes_feat  # TODO: change to concat
        return out


class UNetGNN(nn.Module):
    """ fully convolutional encoder decoder based UNet architecture - With Edge Collapse based pooling"""

    def __init__(self, ncf, num_segmentation_classes=10):
        super(UNetGNN, self).__init__()
        self.current_net = None
        ncf = ncf[::-1]
        for i in range(1, len(ncf)):
            self.current_net = UNetBlock(ncf[i], ncf[i - 1], self.current_net)

        self.lin = nn.Linear(ncf[-1], num_segmentation_classes)

    def forward(self, nodes_feat, edges_index, batch):
        out = self.current_net(nodes_feat, edges_index, batch)
        out = self.lin(out)
        return out

# test
if __name__ == '__main__':
    edges_num = 1000
    vertices_num = 100
    test_nodes_features = torch.rand((vertices_num, 10))
    test_edges_index = torch.ceil(torch.rand((2, edges_num)) * (vertices_num - 1)).type(torch.LongTensor)
    data = Data(x=test_nodes_features, edge_index=test_edges_index)
    batch = Batch.from_data_list([data])

    unet = UNetGNN([10, 20, 20, 20])
    print(unet)
    unet(test_nodes_features, test_edges_index, batch.batch)
