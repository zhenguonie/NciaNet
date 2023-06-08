import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import pgl
import os.path
import sys
from pgl.utils.helper import generate_segment_id_from_index
from pgl.utils import op
sys.path.append(os.path.dirname(os.path.dirname(r'C:/Users/user/Desktop/layers.py')))
from Desktop.layers import van_layer, Hbond_layer, hyb_layer,PAI_layer,gated_skip_connection,output_layer2,get_diepersed_node_feature,SpatialInputLayer, MyBond2AtomLayer,Atom2BondLayer, my_Atom2BondLayer ,Bond2BondLayer, Bond2AtomLayer, PiPoolLayer, OutputLayer
from pgl.nn import GATConv

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


class NciaNet(nn.Layer):
    def __init__(self, args):
        super(NciaNet, self).__init__()
        num_convs = args.num_convs
        dense_dims = args.dense_dims
        hidden_dim = args.hidden_dim
        self.num_convs = num_convs
        cut_dist = args.cut_dist
        num_angle = args.num_angle
        activation = args.activation
        feat_drop = args.feat_drop

        self.input_layer = SpatialInputLayer(hidden_dim, cut_dist, activation=F.relu)
        self.get_diepersed_node_feature= get_diepersed_node_feature()
        self.my_atom2bondlayer = nn.LayerList()
        self.atom2bond_layers = nn.LayerList()
        self.bond2bond_layers = nn.LayerList()
        self.bond2atom_layers = nn.LayerList()
        self.gated_skip_connection = nn.LayerList()
        self.MyBond2AtomLayer = nn.LayerList()
        self.GATLayer= nn.LayerList()

        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 36, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        self.atom2bond_layers.append(Atom2BondLayer(atom_dim= 128, bond_dim= 128, activation=activation))
        
        self.gated_skip_connection.append(gated_skip_connection(128 , 0.2))
        self.gated_skip_connection.append(gated_skip_connection(128 , 0.2))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , 0.2))
        self.gated_skip_connection.append(gated_skip_connection(128 , 0.2))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        self.gated_skip_connection.append(gated_skip_connection(128 , feat_drop))
        
        self.MyBond2AtomLayer.append(MyBond2AtomLayer(36 ,hidden_dim ,feat_drop,activation=None))
        self.MyBond2AtomLayer.append(MyBond2AtomLayer(128 ,hidden_dim ,feat_drop, activation=None))
        self.MyBond2AtomLayer.append(MyBond2AtomLayer(128 ,hidden_dim ,feat_drop, activation=None))
        self.MyBond2AtomLayer.append(MyBond2AtomLayer(128 ,hidden_dim ,feat_drop, activation=None))
        self.MyBond2AtomLayer.append(MyBond2AtomLayer(128 ,hidden_dim ,feat_drop,activation=None))
        self.MyBond2AtomLayer.append(MyBond2AtomLayer(128 ,hidden_dim ,feat_drop, activation=None))
        self.MyBond2AtomLayer.append(MyBond2AtomLayer(128 ,hidden_dim ,feat_drop,activation=None))
        self.MyBond2AtomLayer.append(MyBond2AtomLayer(128 ,hidden_dim ,feat_drop, activation=None))
        
        self.GATLayer.append(GATConv(input_size=36 ,hidden_size=128 ,feat_drop=0.0,attn_drop=0.0, num_heads=1,concat=False,activation=F.relu))
        self.GATLayer.append(GATConv(input_size=128 ,hidden_size=128 ,feat_drop=0.0,attn_drop=0.0, num_heads=1,concat=False,activation=F.relu))
        self.GATLayer.append(GATConv(input_size=128 ,hidden_size=128 ,feat_drop=0.0,attn_drop=0.0, num_heads=1,concat=False,activation=F.relu))
             
        self.Hbond_layer = Hbond_layer(128,0.2)
        self.hyb_layer = hyb_layer(128,0.2)
        self.van_layer= van_layer(128,0.2)
        self.PAI_layer= PAI_layer(128,0.2)
             
        self.pipool_layer = PiPoolLayer(hidden_dim, hidden_dim, num_angle)
        self.output_layer = OutputLayer(128*3, dense_dims,feat_drop)
        self.output_layer2 = output_layer2(128 *2, dense_dims,0.2)
        self.input = None
        self.final = None
        self.final_conv_grads =None
        self.pool = pgl.nn.GraphPool(pool_type='sum')
        self.output = nn.Linear(128, 1)
        self.fc_sim = nn.Linear(128,3)
        self.relu = F.relu
        self.feat_drop2 = nn.Dropout(p=0.2)
        middle_dim = 128
        self.mlp = nn.LayerList()
        for hidden_dim in dense_dims:
            self.mlp.append(DenseLayer(middle_dim, hidden_dim, activation=F.relu, bias=True))
            middle_dim = hidden_dim
            
    def activations_hook(self, grad):
        y_grad = list()
        y_grad.append(grad)

    def forward(self, a2a_g, Hbond_graphs, hyb_graphs,van_graphs,PAI_graphs):
        atom_feat = a2a_g.node_feat['feat']
        dist_feat = a2a_g.edge_feat['dist']
        atom_feat = paddle.cast(atom_feat, 'float32')
        dist_feat = paddle.cast(dist_feat, 'float32')
        hbond_feature = Hbond_graphs.edge_feat['dist']
        hbond_coe = Hbond_graphs.edge_feat['coe']
        hbond_coe2 = Hbond_graphs.node_feat['coe']
        van_feature=van_graphs.edge_feat['dist']
        rep_att = van_graphs.edge_feat['rep_att']
        atom_h=atom_feat
        dist_h = self.input_layer(dist_feat)
        bond_h0 = self.atom2bond_layers[0](a2a_g, atom_h, dist_h)
        atom_h0 = self.MyBond2AtomLayer[0](a2a_g,atom_h, bond_h0)
        bond_h1 = self.atom2bond_layers[1](a2a_g, atom_h0, dist_h)
        atom_h1 = self.MyBond2AtomLayer[1](a2a_g,atom_h0, bond_h1)
        atom_h1 = self.gated_skip_connection[1](atom_h0, atom_h1) 
        bond_h2 = self.atom2bond_layers[2](a2a_g, atom_h1, dist_h)
        atom_h2 = self.MyBond2AtomLayer[2](a2a_g,atom_h1, bond_h2)
        atom_h2 = self.gated_skip_connection[2](atom_h1, atom_h2) 
        bond_h3 = self.atom2bond_layers[3](a2a_g, atom_h2, dist_h)
        atom_h3 = self.MyBond2AtomLayer[3](a2a_g,atom_h2, bond_h3)
        hbond = self.Hbond_layer(Hbond_graphs ,atom_h3,hbond_feature, hbond_coe,hbond_coe2)
        van = self.van_layer(van_graphs,atom_h3 ,van_feature,rep_att)     
        concanted= paddle.concat(x=[hbond,van],axis=-1)  
        pred_socre = self.output_layer2(a2a_g, concanted)
        return pred_socre