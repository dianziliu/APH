import torch
import torch.nn as nn
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, sort_edge_index

from torch.nn import Linear

# scatter_, 1.5 以后被取消了，有torch.scatter 代替


class GraphPool(nn.Module):
    def __init__(self, hidd_dim, ratio=0.9, non_linearity=torch.sigmoid):
        super().__init__()
        self.ratio = ratio
        self.non_linearity = non_linearity
        self.vec1 = nn.Parameter(torch.zeros(
            1, hidd_dim), requires_grad=True)
        init.xavier_uniform_(self.vec1.data,0.1)

    def forward(self, ui_em, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # (batch * word_num)
        scores1 = torch.abs(torch.sum(ui_em[batch] * x, 1))
        # scores2 = self.dist(x, edge_index)
        # scores = scores1 + scores2
        scores=scores1
        perm = topk(scores, self.ratio, batch)
        x = x[perm] * self.non_linearity(scores[perm]).view(-1, 1)
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=scores.size(0))
        batch = batch[perm]
        return x, edge_index, edge_attr, batch, perm

    def dist(self, x, edge_index):
        edge_index_sort, _ = sort_edge_index(edge_index, num_nodes=x.size(0))
        dis_em = torch.abs(
            (x[edge_index_sort[0]] * self.vec1).sum(-1) - (x[edge_index_sort[1]] * self.vec1).sum(-1))
        # dis_em = scatter_('mean', dis_em.unsqueeze(
        #     1), edge_index_sort[0])  # (word_num, dim)
        # dis_em=torch.scatter(dis_em,0,edge_index_sort[0],dis_em,reduce="add")
        # dis_em = torch.scatter('mean',dis_em.unsqueeze(1), edge_index_sort[0])  # (word_num, dim)       
        dis_em2=dis_em.clone()
        return dis_em.scatter_(0,edge_index_sort[0],dis_em2,reduce="add")


class GAT(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relation,node_embed,idx2edge, 
                 negative_slope=0.2, dropout=0, **kwargs):
        super(GAT, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        #
        self.node_embed=node_embed  # 节点的嵌入表
        self.idx2edge=idx2edge      # 边索引的转化信息
        self.relation_w = nn.Parameter(torch.Tensor(
            num_relation, in_channels))
        self.weight = Parameter(
            torch.Tensor(in_channels, out_channels))
        self.weight1 = Parameter(
            torch.Tensor(in_channels, out_channels))
        self.weight2 = Parameter(
            torch.Tensor(in_channels, out_channels))
        self.weight3 = Parameter(
            torch.Tensor(in_channels, out_channels))

        init.xavier_uniform_(self.relation_w.data,0.1)
        init.xavier_uniform_(self.weight.data,0.1)
        init.xavier_uniform_(self.weight1.data,0.1)
        init.xavier_uniform_(self.weight2.data,0.1)
        init.xavier_uniform_(self.weight3.data,0.1)

    def forward(self, x, batch, edge_index, edge_type, size=None):
        return self.propagate(edge_index, size=size, x=x, batch=batch, edge_type=edge_type)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_type):
        edges= self.idx2edge[edge_type]
        # 
        rel_emb = torch.index_select(self.relation_w, 0, edges[:,0])
        
        tail_emb=self.node_embed(edges[:,1])
        
        x_iw = torch.einsum("ij,jk->ik",x_i,self.weight2)
        x_jw = torch.einsum("ij,jk->ik",x_j,self.weight3)
        x_r = torch.einsum("ij,jk->ik",rel_emb,self.weight1)
        # x_iw = torch.matmul(x_i, self.weight2)
        # x_jw = torch.matmul(x_j, self.weight3)
        # x_r = torch.matmul(rel_emb, self.weight1)
        # torch.matmul(x_j, self.weight)
        
        
        alpha = (x_iw * (x_jw + x_r)).sum(-1)
        
        # 对应公式（2）
        alpha = F.leaky_relu(alpha, self.negative_slope)
        # alpha = softmax(alpha,edge_index_i,size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return torch.einsum("ij,jk->ik",x_j,self.weight) * alpha.view(-1, 1)

    def update(self, aggr_out, x):
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.channels, self.channels)


class GAT2(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relation,node_embed,idx2edge, 
                 negative_slope=0.2, dropout=0, **kwargs):
        super(GAT2, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        #
        self.node_embed=node_embed  # 节点的嵌入表
        self.idx2edge=idx2edge      # 边索引的转化信息
        self.relation_w = nn.Parameter(torch.Tensor(
            num_relation, in_channels))
        self.weight = Parameter(
            torch.Tensor(in_channels, out_channels))
        self.weight1 = Parameter(
            torch.Tensor(in_channels, out_channels))
        self.weight2 = Parameter(
            torch.Tensor(in_channels, out_channels))
        self.weight3 = Parameter(
            torch.Tensor(in_channels, out_channels))

        self.mlp1 =Linear(2*in_channels,in_channels)
        self.mlp2 =Linear(2*in_channels,in_channels)

        init.xavier_uniform_(self.relation_w.data,0.1)
        init.xavier_uniform_(self.weight.data,0.1)
        init.xavier_uniform_(self.weight1.data,0.1)
        init.xavier_uniform_(self.weight2.data,0.1)
        init.xavier_uniform_(self.weight3.data,0.1)

    def forward(self, x, batch, edge_index, edge_type, size=None):
        return self.propagate(edge_index, size=size, x=x, batch=batch, edge_type=edge_type)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_type):
        edges= self.idx2edge[edge_type]
        # 
        rel_emb = torch.index_select(self.relation_w, 0, edges[:,0])
        
        ui_emb=self.node_embed(edges[:,1])
        
        r_emb=self.mlp1(torch.concat([rel_emb,ui_emb],dim=1))
        tail_emb=self.mlp2(torch.concat([x_j,ui_emb],dim=1))
        
        x_iw = torch.einsum("ij,jk->ik",x_i,self.weight2)
        x_jw = torch.einsum("ij,jk->ik",tail_emb,self.weight3)
        x_r = torch.einsum("ij,jk->ik",r_emb,self.weight1)
        # x_iw = torch.matmul(x_i, self.weight2)
        # x_jw = torch.matmul(x_j, self.weight3)
        # x_r = torch.matmul(rel_emb, self.weight1)
        # torch.matmul(x_j, self.weight)
        
        
        alpha = (x_iw * (x_jw + x_r)).sum(-1)
        
        # 对应公式（2）
        alpha = F.leaky_relu(alpha, self.negative_slope)
        # alpha = softmax(alpha,edge_index_i,size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return torch.einsum("ij,jk->ik",x_j,self.weight) * alpha.view(-1, 1)

    def update(self, aggr_out, x):
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.channels, self.channels)

