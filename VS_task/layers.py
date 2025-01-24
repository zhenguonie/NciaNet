import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from pgl.utils.helper import generate_segment_id_from_index
from pgl.utils import op
import pgl.math as math
from NciaNet.VS_task.utils import generate_segment_id



class DenseLayer(nn.Layer):
    def __init__(self, in_dim, out_dim, activation=F.relu, bias=True):
        super(DenseLayer, self).__init__()
        self.activation = activation
        if not bias:
            self.fc = nn.Linear(in_dim, out_dim, bias_attr=False)
        else:
            self.fc = nn.Linear(in_dim, out_dim)
    
    def forward(self, input_feat):
        return self.activation(self.fc(input_feat))
    
class gated_skip_connection(nn.Layer):
    
    
    def __init__(self, hidden_dim, dropout):
        super(gated_skip_connection, self).__init__()
        self.fc = nn.Linear(hidden_dim*2, hidden_dim, bias_attr=True)
        self.sigmoid = nn.Sigmoid()
        self.feat_drop = nn.Dropout(p=dropout)
        
    def forward(self, atom_feat, atom_feat2):
         atom_feat = self.feat_drop(atom_feat)
         atom_feat2 = self.feat_drop(atom_feat2)
         atom_feat3 = paddle.concat([atom_feat, atom_feat2], axis=-1)
         atom_feat3 = self.fc(atom_feat3)
         coe = self.sigmoid(atom_feat3)
         atom_feat2 = paddle.multiply(coe, atom_feat2)+paddle.multiply(1-coe, atom_feat)
         return atom_feat2

class SpatialInputLayer(nn.Layer):
    """Implementation of Spatial Relation Embedding Module.
    """
    def __init__(self, hidden_dim, cut_dist, activation=F.relu):
        super(SpatialInputLayer, self).__init__()
        self.cut_dist = cut_dist
        self.dist_embedding_layer = nn.Embedding(int(cut_dist)-1, hidden_dim, sparse=True)
        self.dist_input_layer = DenseLayer(hidden_dim, hidden_dim, activation, bias=True)
    
    def forward(self, dist_feat):
        dist_feat = dist_feat.squeeze()
        idx = paddle.nonzero(dist_feat)
        dist_feat = dist_feat[idx]
        dist = paddle.clip(dist_feat.squeeze(), 1.0, self.cut_dist-1e-6).astype('int64') - 1
        eh_emb = self.dist_embedding_layer(dist)
        eh_emb = self.dist_input_layer(eh_emb)
        return eh_emb


class Atom2BondLayer(nn.Layer):
    """Implementation of Node->Edge Aggregation Layer.
    """
    def __init__(self, atom_dim, bond_dim, activation=F.relu):
        super(Atom2BondLayer, self).__init__()
        in_dim = atom_dim*2 + bond_dim
        self.fc_agg = DenseLayer(in_dim, bond_dim, activation=activation, bias=True)
        
    def agg_func(self, src_feat, dst_feat, edge_feat):
        h_src = src_feat['h']
        h_dst = dst_feat['h']
        h_agg = paddle.concat([h_src, h_dst, edge_feat['h']], axis=-1)
        return {'h': h_agg}

    def forward(self, g, atom_feat, edge_feat):
        msg = g.send(self.agg_func, src_feat={'h': atom_feat}, dst_feat={'h': atom_feat}, edge_feat={'h': edge_feat})
        bond_feat = msg['h']
        bond_feat = self.fc_agg(bond_feat)
        return bond_feat
    
    
class MyBond2AtomLayer(nn.Layer):
    """Implementation of Angle-oriented Edge->Edge Aggregation Layer.
    """
    def __init__(self, atom_dim, hidden_dim,  dropout, activation=None):
        super(MyBond2AtomLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        self.conv_layer = DomainAttentionLayer(atom_dim, self.hidden_dim, dropout, activation=None)
        self.activation = activation
    
    def forward(self, g,atom_h, bond_feat):
        feat_h = self.conv_layer(g, atom_h,bond_feat)
        if self.activation:
            feat_h = self.activation(feat_h)
        return feat_h

class DomainAttentionLayer(nn.Layer):
    """Implementation of Angle Domain-speicific Attention Layer.
    """
    def __init__(self, atom_dim, hidden_dim, dropout, activation=F.relu):
        super(DomainAttentionLayer, self).__init__()
        self.attn_fc = nn.Linear(atom_dim*2, 4)
        self.attn_out = nn.Linear(3, 1, bias_attr=False)
        self.feat_drop = nn.Dropout(p=dropout)
        self.attn_drop = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.relu = F.relu
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.activation = activation
        self.softamx = paddle.nn.Softmax(axis = 0)
    
    def attn_send_func(self, src_feat, dst_feat, edge_feat):
        h_c = paddle.concat([src_feat['h']  , dst_feat['h']], axis=-1)
        h_c = self.attn_fc(h_c)
        h_c = self.leaky_relu(h_c)
        return {"alpha": h_c, "h": edge_feat["h"]}
    
    def attn_recv_func(self, msg):
        alpha = msg.reduce_softmax(msg["alpha"])
        alpha = paddle.mean(alpha, axis=-1, keepdim=True)
        alpha = self.attn_drop(alpha) 
        feature = msg["h"] 
        feature = feature * alpha
        feature = msg.reduce(feature, pool_type="sum")
        return feature

    def forward(self, g, atom_h, bond_feat):
        bond_feat = self.feat_drop(bond_feat)
        msg = g.send(self.attn_send_func,
                    src_feat={"h": atom_h},
                    dst_feat={"h": atom_h},
                    edge_feat={"h": bond_feat})
        rst = g.recv(reduce_func=self.attn_recv_func, msg=msg)
        if self.activation:
            rst = self.activation(rst)
        return rst




class Bond2AtomLayer(nn.Layer):
    """Implementation of Distance-aware Edge->Node Aggregation Layer.
    """
    def __init__(self, bond_dim, atom_dim, hidden_dim, num_heads, dropout, merge='mean', activation=F.relu):
        super(Bond2AtomLayer, self).__init__()
        self.merge = merge
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.src_fc = nn.Linear(bond_dim, num_heads * hidden_dim)
        self.dst_fc = nn.Linear(atom_dim, num_heads * hidden_dim)
        self.edg_fc = nn.Linear(hidden_dim, num_heads * hidden_dim)
        self.weight_src = self.create_parameter(shape=[num_heads, hidden_dim])
        self.weight_dst = self.create_parameter(shape=[num_heads, hidden_dim])
        self.weight_edg = self.create_parameter(shape=[num_heads, hidden_dim])
        
        self.feat_drop = nn.Dropout(p=dropout)
        self.attn_drop = nn.Dropout(p=dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.activation = activation

    def attn_send_func(self, src_feat, dst_feat, edge_feat):
        alpha = src_feat["attn"] + dst_feat["attn"] + edge_feat['attn']
        alpha = self.leaky_relu(alpha)
        return {"alpha": alpha, "h": src_feat["h"]}
    
    def attn_recv_func(self, msg):
        alpha = msg.reduce_softmax(msg["alpha"])
        alpha = paddle.reshape(alpha, [-1, self.num_heads, 1])
        alpha = self.attn_drop(alpha)

        feature = msg["h"]
        feature = paddle.reshape(feature, [-1, self.num_heads, self.hidden_dim])
        feature = feature * alpha
        if self.merge == 'cat':
            feature = paddle.reshape(feature, [-1, self.num_heads * self.hidden_dim])
        if self.merge == 'mean':
            feature = paddle.mean(feature, axis=1)

        feature = msg.reduce(feature, pool_type="sum")
        return feature

    def forward(self, g, atom_feat, bond_feat, edge_feat):
        bond_feat = self.feat_drop(bond_feat)
        atom_feat = self.feat_drop(atom_feat)
        edge_feat = self.feat_drop(edge_feat)

        bond_feat = self.src_fc(bond_feat)
        atom_feat = self.dst_fc(atom_feat)
        edge_feat = self.edg_fc(edge_feat)
        bond_feat = paddle.reshape(bond_feat, [-1, self.num_heads, self.hidden_dim])
        atom_feat = paddle.reshape(atom_feat, [-1, self.num_heads, self.hidden_dim])
        edge_feat = paddle.reshape(edge_feat, [-1, self.num_heads, self.hidden_dim])

        attn_src = paddle.sum(bond_feat * self.weight_src, axis=-1)
        attn_dst = paddle.sum(atom_feat * self.weight_dst, axis=-1)
        attn_edg = paddle.sum(edge_feat * self.weight_edg, axis=-1)

        msg = g.send(self.attn_send_func,
                     src_feat={"attn": attn_src, "h": bond_feat},
                     dst_feat={"attn": attn_dst},
                     edge_feat={'attn': attn_edg})
        rst = g.recv(reduce_func=self.attn_recv_func, msg=msg)

        if self.activation:
            rst = self.activation(rst)
        return rst



    
    
class get_diepersed_edge_feature(nn.Layer):
    """Implementation of Node->Edge Aggregation Layer.
    """
    #两点之间的边进行信息聚合
    def __init__(self):
        super(get_diepersed_edge_feature, self).__init__()
#        self.batch_size = batch_size
#        self.eh_emb = eh_emb

    def get_graph_edge_conunt(self,a2a_g , batch_size):
        batched_Graph_edge_id_list = a2a_g.graph_edge_id
        batched_Graph_edge_id_list=batched_Graph_edge_id_list.numpy().tolist()
        edge_count_dict = {}
        for key in batched_Graph_edge_id_list:        
            edge_count_dict[key] = edge_count_dict.get(key,0)+1
        return edge_count_dict

    def forward(self, g, eh_emb, batch_size):
        edge_count_dict = self.get_graph_edge_conunt(g , batch_size)
        feature_matrix =[[] for _ in range(batch_size)]
        feature_idx = 0
        for key in edge_count_dict:
            feature_idx2 = edge_count_dict[key]
            feature_idx =+ feature_idx+feature_idx2
            feature_idx3 = feature_idx-feature_idx2
            feature_matrix[key].append(eh_emb [feature_idx3:feature_idx,:])
        return feature_matrix

class my_Atom2BondLayer(nn.Layer):
    """Implementation of Angle-oriented Edge->Edge Aggregation Layer.
    """
    def __init__(self,concated_dim , atom_dim, hidden_dim, cut_dist, dropout, merge='cat', activation=None):
        #
        super(my_Atom2BondLayer, self).__init__()
        self.intergrate = DenseLayer(concated_dim, hidden_dim, activation=F.relu, bias=True)
        self.cut_dist = cut_dist
        self.hidden_dim = hidden_dim
        self.merge = merge
        self.conv_layer = nn.LayerList()
        self.get_diepersed_edge_feature = get_diepersed_edge_feature()
        for _ in range(int(cut_dist-1)):
            conv = DomainAttentionLayer(atom_dim, hidden_dim, dropout, activation=None)
            self.conv_layer.append(conv)
        self.activation = activation

    def get_edge_idx(self, b2a_gl, k):
        b2a_g_edge_id = b2a_gl[k].graph_edge_id.numpy().tolist()
        edge_count_dict = {}
        for key in b2a_g_edge_id:
            edge_count_dict[key] = edge_count_dict.get(key,0)+1
        return edge_count_dict

        
    
    def get_bond_idx(self, b2a_gl, k,edge_count_dict):
        b2a_g_edge_order_idx = b2a_gl[k].edge_feat["idx"].numpy().tolist()
        edge_order_idx_matrix = []
        feature_idx = 0
        for key in edge_count_dict:
            feature_idx2 = edge_count_dict[key]
            feature_idx =+ feature_idx+feature_idx2
            feature_idx3 = feature_idx-feature_idx2
            edge_order_idx_matrix.append(b2a_g_edge_order_idx[feature_idx3:feature_idx])
        return edge_order_idx_matrix

    def get_bond_feat(self, edge_order_idx_matrix, bond_feat, batch_size):
        bond_feats = []
        for i in range(batch_size):
            edge_order_idx_matrix1 = edge_order_idx_matrix[i]
            embedding_matrix = bond_feat[i][0]
            for i in range(len(edge_order_idx_matrix1)):
                idx = edge_order_idx_matrix1[i]
                edge_id_feat = embedding_matrix[idx]
                edge_id_feat1 = paddle.reshape(edge_id_feat, [1, -1])
                bond_feats.append(edge_id_feat1)
        edge_id_feat1 = paddle.concat(bond_feats, axis=0)
        return edge_id_feat1
    
    def forward(self,a2a_g, bond_h0 , g_list, atom_h, batch_size):
        feature_matrix = self.get_diepersed_edge_feature(a2a_g , bond_h0 , batch_size)
        h_list = []
        for k in range(int(self.cut_dist-1)):
            edge_count_dict = self.get_edge_idx(g_list, k)
            edge_order_idx_matrix = self.get_bond_idx(g_list, k,edge_count_dict)
            edge_id_feat1 = self.get_bond_feat(edge_order_idx_matrix ,feature_matrix, batch_size)
            h = self.conv_layer[k](g_list[k], atom_h,edge_id_feat1)
            h_list.append(h)

        if self.merge == 'cat':
            feat_h = paddle.concat(h_list, axis=-1)
            feat_h = self.intergrate(feat_h)
        if self.merge == 'mean':
            feat_h = paddle.mean(paddle.stack(h_list, axis=-1), axis=1)
        if self.merge == 'sum':
            feat_h = paddle.sum(paddle.stack(h_list, axis=-1), axis=1)
        if self.merge == 'max':
            feat_h = paddle.max(paddle.stack(h_list, axis=-1), axis=1)
        if self.merge == 'cat_max':
            feat_h = paddle.stack(h_list, axis=-1)
            feat_max = paddle.max(feat_h, dim=1)[0]
            feat_max = paddle.reshape(feat_max, [-1, 1, self.hidden_dim])
            feat_h = paddle.reshape(feat_h * feat_max, [-1, int(self.cut_dist-1) * self.hidden_dim])

        if self.activation:
            feat_h = self.activation(feat_h)
        return feat_h
    
    
    
class Bond2BondLayer(nn.Layer):
    def __init__(self, bond_dim, hidden_dim, num_angle, dropout, merge='cat', activation=None):
        super(Bond2BondLayer, self).__init__()
        self.num_angle = num_angle
        self.hidden_dim = hidden_dim
        self.merge = merge
        self.conv_layer = nn.LayerList()
        for _ in range(num_angle):
            conv = DomainAttentionLayer(bond_dim, hidden_dim, dropout, activation=None)
            self.conv_layer.append(conv)
        self.activation = activation
    
    def forward(self, g_list, bond_feat):
        h_list = []
        for k in range(self.num_angle):
            h = self.conv_layer[k](g_list[k], bond_feat)
            h_list.append(h)

        if self.merge == 'cat':
            feat_h = paddle.concat(h_list, axis=-1)
        if self.merge == 'mean':
            feat_h = paddle.mean(paddle.stack(h_list, axis=-1), axis=1)
        if self.merge == 'sum':
            feat_h = paddle.sum(paddle.stack(h_list, axis=-1), axis=1)
        if self.merge == 'max':
            feat_h = paddle.max(paddle.stack(h_list, axis=-1), axis=1)
        if self.merge == 'cat_max':
            feat_h = paddle.stack(h_list, axis=-1)
            feat_max = paddle.max(feat_h, dim=1)[0]
            feat_max = paddle.reshape(feat_max, [-1, 1, self.hidden_dim])
            feat_h = paddle.reshape(feat_h * feat_max, [-1, self.num_angle * self.hidden_dim])

        if self.activation:
            feat_h = self.activation(feat_h)
        return feat_h
    

class PiPoolLayer(nn.Layer):
    """Implementation of Pairwise Interactive Pooling Layer.
    """
    def __init__(self, bond_dim, hidden_dim, num_angle):
        super(PiPoolLayer, self).__init__()
        self.bond_dim = bond_dim
        self.num_angle = num_angle
        self.num_type = 4 * 9
        fc_in_dim = num_angle * bond_dim
        self.fc_1 = DenseLayer(fc_in_dim, hidden_dim, activation=F.relu, bias=True)
        self.fc_2 = nn.Linear(hidden_dim, 1, bias_attr=False)
        self.softmax = nn.Softmax(axis=1)
    
    def forward(self, bond_types_batch, type_count_batch, bond_feat):
        """
        Input example:
            bond_types_batch: [0,0,2,0,1,2] + [0,0,2,0,1,2] + [2]
            type_count_batch: [[3, 3, 0], [1, 1, 0], [2, 2, 1]] # [num_type, batch_size]
        """
        bond_feat = self.fc_1(paddle.reshape(bond_feat, [-1, self.num_angle*self.bond_dim]))
        inter_mat_list =[]
        for type_i in range(self.num_type):
            type_i_index = paddle.masked_select(paddle.arange(len(bond_feat)), bond_types_batch==type_i)
            if paddle.sum(type_count_batch[type_i]) == 0:
                inter_mat_list.append(paddle.to_tensor(np.array([0.]*len(type_count_batch[type_i])), dtype='float32'))
                continue
            bond_feat_type_i = paddle.gather(bond_feat, type_i_index)
            graph_bond_index = op.get_index_from_counts(type_count_batch[type_i])
            graph_bond_id = generate_segment_id(graph_bond_index)
            graph_feat_type_i = math.segment_pool(bond_feat_type_i, graph_bond_id, pool_type='sum')
            mat_flat_type_i = self.fc_2(graph_feat_type_i).squeeze(1)
            my_pad = nn.Pad1D(padding=[0, len(type_count_batch[type_i])-len(mat_flat_type_i)], value=-1e9)
            mat_flat_type_i = my_pad(mat_flat_type_i)
            inter_mat_list.append(mat_flat_type_i)

        inter_mat_batch = paddle.stack(inter_mat_list, axis=1) # [batch_size, num_type]
        inter_mat_mask = paddle.ones_like(inter_mat_batch) * -1e9
        inter_mat_batch = paddle.where(type_count_batch.transpose([1, 0])>0, inter_mat_batch, inter_mat_mask)
        inter_mat_batch = self.softmax(inter_mat_batch)
        return inter_mat_batch


class OutputLayer(nn.Layer):
    """Implementation of Prediction Layer.
    """
    def __init__(self, atom_dim, hidden_dim_list ,dropout):
        super(OutputLayer, self).__init__()
        self.poo0 = pgl.nn.GraphPool(pool_type='mean')
        self.pool = pgl.nn.GraphPool(pool_type='sum')
        self.mlp = nn.LayerList()
        for hidden_dim in hidden_dim_list:
            self.mlp.append(DenseLayer(atom_dim, hidden_dim, activation=F.relu))
            atom_dim = hidden_dim
        self.output_layer = nn.Linear(atom_dim, 1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.get_diepersed_node_feature = get_diepersed_node_feature()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.relu = F.relu
        self.sigmoid = F.sigmoid
        self.elu = F.elu
        self.softamx = paddle.nn.Softmax()
        self.feat_drop = nn.Dropout(p=dropout)
    
    def forward(self,  hbond,hyb,van):
        hbond1=self.fc1(hbond)
        hbond = self.relu(hbond1)
        hyb1=self.fc2(hyb)
        hyb = self.relu(hyb1)
        van1=self.fc3(van)
        van = self.relu(van1)
        concanted= paddle.concat(x=[hbond,hyb,van],axis=-1)
        for layer in self.mlp:
            concanted = layer(concanted)
        output = self.output_layer(concanted)
        return output

class get_diepersed_node_feature(nn.Layer):
    def __init__(self):
        super(get_diepersed_node_feature, self).__init__()

    def get_graph_node_conunt(self,a2a_g , batch_size):
        batched_Graph_node_id_list = a2a_g.graph_node_id
        batched_Graph_node_id_list=batched_Graph_node_id_list.numpy().tolist()
        node_count_dict = {}
        for key in batched_Graph_node_id_list:        
            node_count_dict[key] = node_count_dict.get(key,0)+1
        return node_count_dict

    def forward(self, g, graph_feat):
        batch_size = len(set(g.graph_node_id.numpy().tolist()))
        edge_count_dict = self.get_graph_node_conunt(g , batch_size)
        feature_matrix =[[] for _ in range(batch_size)]
        feature_idx = 0
        for key in edge_count_dict:
            feature_idx2 = edge_count_dict[key]
            feature_idx =+ feature_idx+feature_idx2
            feature_idx3 = feature_idx-feature_idx2
            feature_matrix[key].append(graph_feat[feature_idx3:feature_idx,:])
        return feature_matrix
    
class output_layer2(nn.Layer):
    """Implementation of Prediction Layer.
    """
    def __init__(self, atom_dim, hidden_dim_list ,dropout):
        super(output_layer2, self).__init__()
        self.poo0 = pgl.nn.GraphPool(pool_type='mean')
        self.pool = pgl.nn.GraphPool(pool_type='sum')
        self.mlp = nn.LayerList()
        for hidden_dim in hidden_dim_list:
            self.mlp.append(DenseLayer(atom_dim, hidden_dim, activation=F.sigmoid))
            atom_dim = hidden_dim
        self.output_layer = nn.Linear(atom_dim, 1)
        self.fc_sim = nn.Linear(128*3, 3)
        self.get_diepersed_node_feature = get_diepersed_node_feature()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.relu = F.relu
        self.sigmoid = F.sigmoid
        self.elu = F.elu
        self.softamx = paddle.nn.Softmax()
        self.feat_drop = nn.Dropout(p=dropout)
    
    def forward(self, g, graph_feat):
        # Pass through MLP layers
        for layer in self.mlp:
            graph_feat = layer(graph_feat)
        
        # Output from last linear layer
        output = self.output_layer(graph_feat)
        
        # Apply sigmoid to ensure output is between 0 and 1
        output = self.sigmoid(output)
        
        return output, graph_feat

 
    
class Hbond_layer(nn.Layer):
    """Implementation of Distance-aware Edge->Node Aggregation Layer.
    """
    def __init__(self, hidden_dim,dropout):
        super(Hbond_layer, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(hidden_dim*2+2, 128)
        self.relu = F.relu
        self.feat_drop = nn.Dropout(p=dropout)
        
    def attn_send_func(self, src_feat, dst_feat, edge_feat):
        edge_feat = self.fc1(edge_feat['angle'])
        edge_feat =self.relu(edge_feat)
        Hbonds = paddle.concat(x=[paddle.multiply(src_feat['h'],src_feat['coe2']),paddle.multiply(dst_feat['h'],dst_feat['coe2']),edge_feat],axis=-1)
        Hbonds = self.fc2(Hbonds)
        Hbonds = self.relu(Hbonds)
        return {'Hbonds': Hbonds}

    def forward(self, g, atom_feat,hbond_feature,hbond_coe,hbond_coe2):
        atom_feat = self.feat_drop(atom_feat)
        hbond_feature = self.feat_drop(hbond_feature)
        msg = g.send(self.attn_send_func,
                     src_feat={'h': atom_feat,'coe2':hbond_coe2},
                     dst_feat={'h': atom_feat,'coe2':hbond_coe2},
                     edge_feat={'angle': hbond_feature,'coe': hbond_coe})
        hbond = math.segment_pool(msg['Hbonds'], g.graph_edge_id,pool_type="sum")
        return hbond


class hyb_layer(nn.Layer):
    """Implementation of Distance-aware Edge->Node Aggregation Layer.
    """
    def __init__(self, hidden_dim,dropout):
        super(hyb_layer, self).__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(hidden_dim*2+1, 128)
        self.relu = F.relu
        self.feat_drop = nn.Dropout(p=dropout)
        
    def attn_send_func(self, src_feat, dst_feat, edge_feat):
        edge_feat = self.fc1(edge_feat['dist'])
        edge_feat =self.relu(edge_feat)
        hybs = paddle.concat(x=[paddle.multiply(src_feat['h'],src_feat['coe2']),paddle.multiply(dst_feat['h'],dst_feat['coe2']),edge_feat],axis=-1)
        hybs = self.fc2(hybs)
        hybs = self.relu(hybs)
        return {'hybs': hybs}

    def forward(self, g, atom_feat,hyb_feature,hyb_coe,hyb_coe2):
        atom_feat = self.feat_drop(atom_feat)
        hyb_feature = self.feat_drop(hyb_feature)
        msg = g.send(self.attn_send_func,
                     src_feat={'h': atom_feat,'coe2':hyb_coe2},
                     dst_feat={'h': atom_feat,'coe2':hyb_coe2},
                     edge_feat={'dist': hyb_feature,'coe': hyb_coe})
        hyb = math.segment_pool(msg['hybs'], g.graph_edge_id,pool_type="sum")
        return hyb



class PAI_layer(nn.Layer):
    """Implementation of Distance-aware Edge->Node Aggregation Layer.
    """
    def __init__(self, hidden_dim,dropout):
        super(PAI_layer, self).__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(hidden_dim*2+1, 128)
        self.relu = F.relu
        self.feat_drop = nn.Dropout(p=dropout)
        
    def attn_send_func(self, src_feat, dst_feat, edge_feat):
        edge_feat = self.fc1(edge_feat['dist'])
        edge_feat =self.relu(edge_feat)
        PAIs = paddle.concat(x=[paddle.multiply(src_feat['h'],src_feat['coe2']),paddle.multiply(dst_feat['h'],dst_feat['coe2']),edge_feat],axis=-1)
        PAIs = self.fc2(PAIs)
        PAIs = self.relu(PAIs)
        return {'hybs': PAIs}

    def forward(self, g, atom_feat,PAI_feature,PAI_coe,PAI_coe2):
        atom_feat = self.feat_drop(atom_feat)
        PAI_feature = self.feat_drop(PAI_feature)
        msg = g.send(self.attn_send_func,
                     src_feat={'h': atom_feat,'coe2':PAI_coe2},
                     dst_feat={'h': atom_feat,'coe2':PAI_coe2},
                     edge_feat={'dist': PAI_feature,'coe': PAI_coe})
        PAI = math.segment_pool(msg['hybs'], g.graph_edge_id,pool_type="sum")
        return PAI
    
class van_layer(nn.Layer):
    """Implementation of Distance-aware Edge->Node Aggregation Layer.
    """
    def __init__(self, hidden_dim,dropout):
        super(van_layer, self).__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(hidden_dim*2+1, 128)
        self.relu = F.relu
        self.feat_drop = nn.Dropout(p=dropout)
        
    def attn_send_func(self, src_feat, dst_feat, edge_feat):
        edge_feat = self.fc1(edge_feat['dist'])
        edge_feat =self.relu(edge_feat)
        vans = paddle.concat(x=[src_feat['h'],dst_feat['h'],edge_feat],axis=-1)
        vans = self.fc2(vans)
        vans = self.relu(vans)
        return {'van': vans}

    def forward(self, g, atom_feat,van_feature,rep_att):
        atom_feat = self.feat_drop(atom_feat)
        van_feature = self.feat_drop(van_feature)
        msg = g.send(self.attn_send_func,
                     src_feat={"h": atom_feat},
                     dst_feat={"h": atom_feat},
                     edge_feat={'dist': van_feature,'rep_att': rep_att})
        van = math.segment_pool(msg['van'], g.graph_edge_id,pool_type="sum")
        return van