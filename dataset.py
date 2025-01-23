"""
Dataset code for protein-ligand complexe interaction graph construction.
"""
import os
import numpy as np
import paddle
import pgl
import pickle
from pgl.utils.data import Dataset as BaseDataset
from pgl.utils.data import Dataloader
from scipy.spatial import distance
from scipy.sparse import coo_matrix
from NciaNet.utils import cos_formula
from tqdm import tqdm
import gc
import random
np.set_printoptions(threshold=np.inf)


class ComplexDataset(BaseDataset):
    def __init__(self, data_path, dataset, cut_dist, save_file=True):
        self.data_path = data_path
        self.dataset = dataset
        self.cut_dist = cut_dist
        self.save_file = save_file
    
        self.labels = []
        self.a2a_graphs = []
        self.hbond_graph= []
        self.hyb_graph= []
        self.van_graph= []
        self.PAI_graph= []
        self.b2a_graphs_list = []

        self.load_data()
        gc.collect()
        

    def __len__(self):
        return len(self.labels)
        gc.collect()
    
    def __getitem__(self, idx):
        return self.a2a_graphs[idx], self.hbond_graph[idx], self.hyb_graph[idx],self.van_graph[idx],self.PAI_graph[idx],\
               self.labels[idx]
        gc.collect()

    def has_cache(self):
        self.graph_path = f'{self.data_path}/{self.dataset}_{int(self.cut_dist)}_pgl_graph.pkl'
        return os.path.exists(self.graph_path)
        gc.collect()
    
    def save(self):
        print('Saving processed complex data...')
        graphs = self.a2a_graphs
        hbond_graphs = self.hbond_graph
        hyb_graphs = self.hyb_graph
        van_graphs =self.van_graph
        PAI_graphs =self.PAI_graph
        with open(self.graph_path, 'wb') as f:
            pickle.dump((graphs,hbond_graphs, hyb_graphs,van_graphs,PAI_graphs,self.labels), f)
        gc.collect()

    def load(self):
        print('Loading processed complex data...')
        with open(self.graph_path, 'rb') as f:
             graphs, hbond_graphs, hyb_graphs, van_graphs,PAI_graphs,labels = pickle.load(f)
        return graphs, hbond_graphs, hyb_graphs,van_graphs,PAI_graphs, labels
        gc.collect()
    
    def build_graph(self, mol):
        num_atoms_d, coords, features, atoms, inter_feats,hbond_idx,hbond_feas,hbond_coe,hyb_ligand,hyb_pocket,name,NEW_DIST,NEW_EDGES, PAI_ligand, PAI_pocket= mol
        concat_edges=[]
        concat_fea=[]
        dist_mat = distance.cdist(coords, coords, 'euclidean')
        np.fill_diagonal(dist_mat, np.inf)
        inter_feats = np.array([inter_feats])
        inter_feats = inter_feats / inter_feats.sum()
        num_atoms = len(coords)
        dist_graph_base = dist_mat.copy()
                   
        for a1 in range(num_atoms_d,num_atoms):
            for a2 in range(num_atoms_d,num_atoms):
                if dist_graph_base[a1][a2] <self.cut_dist and dist_graph_base[a1][a2] != 0:
                   concat_edges.append((a1,a2))
                   concat_fea.append([dist_graph_base[a1][a2]])
                
        n=len(concat_edges)
        
        concat_fea = np.array(concat_fea,dtype=np.float32)
        NEW_DIST= np.array(NEW_DIST,dtype=np.float32).reshape(-1,1)
        NEW_EDGES=NEW_EDGES+concat_edges
        
        NEW_DIST =np.vstack([NEW_DIST, concat_fea])
        assert len(NEW_EDGES) == len(NEW_DIST)
        a2a_graph = pgl.Graph(NEW_EDGES, num_nodes=num_atoms,node_feat={"feat": features}, edge_feat={"dist": NEW_DIST})
        graphs = a2a_graph
        return graphs,n
        gc.collect()

    def build_edge_removing_graph(self, mol):
        num_atoms_d, coords, features, atoms, inter_feats = mol
        dist_mat = distance.cdist(coords, coords, 'euclidean')
        np.fill_diagonal(dist_mat, np.inf)
        num_atoms = len(coords)
        edge_removing_graph = dist_mat.copy()
        for i in range(len(edge_removing_graph)):
            row = edge_removing_graph[i]
            for j in range(len(edge_removing_graph)):
                if row[j]<self.cut_dist:
                    probability = random.uniform(0,1)
                    if probability < 0.2:
                        row[j] = np.inf
        edge_removing_dist_feat = edge_removing_graph[edge_removing_graph < self.cut_dist].reshape(-1,1)
        edge_removing_graph[edge_removing_graph >= self.cut_dist] = 0.
        edge_removed_atom_graph = coo_matrix(edge_removing_graph)
        edge_removed_a2a_edges = list(zip(edge_removed_atom_graph.row, edge_removed_atom_graph.col))
        edge_removed_a2a_graph = pgl.Graph(edge_removed_a2a_edges, num_nodes=num_atoms, node_feat={"feat": features}, edge_feat={"dist": edge_removing_dist_feat})
        graph =edge_removed_a2a_graph
        return graph
     
    
    def build_node_dropping_graph(self, mol):
        num_atoms_d, coords, features, atoms, inter_feats = mol
        dist_mat = distance.cdist(coords, coords, 'euclidean')
        np.fill_diagonal(dist_mat, np.inf)
        node_dropping_graph = dist_mat.copy()
        
        index = []
        for i in range(len(node_dropping_graph)):
            probability = random.uniform(0,1)
            if probability < 0.2:
                index.append(i)
        
        node_dropping_graph = np.delete(node_dropping_graph, index, axis=0)
        node_dropping_graph = np.delete(node_dropping_graph, index, axis=1)
        node_features =  np.delete(features, index, axis=0)
        node_dropping_dist_feat = node_dropping_graph[node_dropping_graph < self.cut_dist].reshape(-1,1)
        num_atoms = len(node_features)
        node_dropping_graph[node_dropping_graph >= self.cut_dist] = 0.
        node_dropping_graph = coo_matrix(node_dropping_graph)
        node_dropping_a2a_edges = list(zip(node_dropping_graph.row, node_dropping_graph.col))
        node_dropping_a2a_graph = pgl.Graph(node_dropping_a2a_edges, num_nodes=num_atoms, node_feat={"feat": node_features}, edge_feat={"dist": node_dropping_dist_feat})
        graph2 =node_dropping_a2a_graph
        return graph2
    
    
    def build_feature_making_graph(self, mol):
        num_atoms_d, coords, features, atoms, inter_feats = mol
        dist_mat = distance.cdist(coords, coords, 'euclidean')
        np.fill_diagonal(dist_mat, np.inf)
        feature_making_graph = dist_mat.copy()
        num_atoms = len(coords)
        features_matrix = features.copy()
        
        for i in range(len(features_matrix)):
            row = features_matrix[i]
            for j in range(36):
                if row[j]!=0:
                    probability = random.uniform(0,1)
                    if probability < 0.2:
                        row[j] = 0
        

        feature_making_dist_feat = feature_making_graph[feature_making_graph < self.cut_dist].reshape(-1,1)
        
        feature_making_graph[feature_making_graph >= self.cut_dist] = 0.
        feature_making_graph = coo_matrix(feature_making_graph)
        feature_making_a2a_edges = list(zip(feature_making_graph.row, feature_making_graph.col))
        feature_making_a2a_graph = pgl.Graph(feature_making_a2a_edges, num_nodes=num_atoms, node_feat={"feat": features_matrix}, edge_feat={"dist": feature_making_dist_feat})
        graph3 =feature_making_a2a_graph
        return graph3
    
    
    
    def making_Hbond_graph(self, mol):
        num_atoms_d, coords, features, atoms, inter_feats, hbond_idx, hbond_feas, hbond_coe, hyb_ligand, hyb_pocket,name,NEW_DIST,NEW_EDGES, PAI_ligand, PAI_pocket= mol
        num_atoms = len(coords)
        hbond_feas= np.array(hbond_feas,dtype=np.float32)
        hbond_coe= np.array(hbond_coe,dtype=np.float32)
        if hbond_idx == [(0, 0)]:
            hbond_coe2 =np.zeros([num_atoms,1],dtype=np.float32)
        else:
            hbond_coe2 =np.ones([num_atoms,1],dtype=np.float32)
        assert hbond_coe2[0]== hbond_coe
        Hbond_graph = pgl.Graph(hbond_idx, num_nodes=num_atoms,node_feat={"coe": hbond_coe2} ,edge_feat={"dist": hbond_feas, "coe": hbond_coe})
        return Hbond_graph


    def making_hyb_graph(self, mol):
        num_atoms_d, coords, features, atoms, inter_feats, hbond_idx, hbond_feas, hbond_coe, hyb_ligand, hyb_pocket,name,NEW_DIST,NEW_EDGES, PAI_ligand, PAI_pocket= mol
        hyb_concat_edges=[]
        hyb_concat_fea=[]
        hyb_cof= []
        num_atoms = len(coords)
        dist_mat = distance.cdist(coords, coords, 'euclidean')
        np.fill_diagonal(dist_mat, np.inf)
        new_distance_mat = dist_mat.copy()
        if len(hyb_ligand)!=0 and len(hyb_pocket)!= 0:
            for a1 in hyb_ligand:
                assert atoms[a1]==6 or atoms[a1]==16 or atoms[a1]==17 or atoms[a1]==35 or atoms[a1]==53 
                for a2 in hyb_pocket:
                    assert atoms[a2]==6 or atoms[a2]==16
                    hyb_concat= new_distance_mat[a1][a2]
                    if hyb_concat< 4.0:
                        hyb_concat_edges.append((a1,a2))
                        hyb_concat_fea.append([hyb_concat])
        assert len(hyb_concat_edges) == len(hyb_concat_fea)
        if len(hyb_concat_edges)==0:
           hyb_concat_edges.append((0,0))
           hyb_concat_fea.append([0.])
           hyb_cof.append([0])
           hyb_concat_fea = np.array(hyb_concat_fea,dtype=np.float32)
           hyb_cof = np.array(hyb_cof,dtype=np.float32)
        else:
            hyb_concat_fea = np.array(hyb_concat_fea,dtype=np.float32).reshape(-1,1)
            hyb_cof.append([1])
            hyb_cof = np.array(hyb_cof,dtype=np.float32)
            
        if hyb_concat_edges == [(0, 0)]:
            hyb_coe2 =np.zeros([num_atoms,1],dtype=np.float32)
        else:
            hyb_coe2 =np.ones([num_atoms,1],dtype=np.float32)
        assert hyb_coe2[0]== hyb_cof
        hyb_graph = pgl.Graph(hyb_concat_edges, num_nodes=num_atoms,node_feat={"coe": hyb_coe2} ,edge_feat={"dist": hyb_concat_fea, "coe": hyb_cof})
        return hyb_graph


    def making_PAI_graph(self, mol):
        num_atoms_d, coords, features, atoms, inter_feats, hbond_idx, hbond_feas, hbond_coe, hyb_ligand, hyb_pocket,name,NEW_DIST,NEW_EDGES, PAI_ligand, PAI_pocket= mol
        PAI_concat_edges=[]
        PAI_concat_fea=[]
        PAI_cof= []
        num_atoms = len(coords)
        dist_mat = distance.cdist(coords, coords, 'euclidean')
        np.fill_diagonal(dist_mat, np.inf)
        new_distance_mat = dist_mat.copy()
        if len(PAI_ligand)!=0 and len(PAI_pocket)!= 0:
            for a1 in PAI_ligand:
                
                for a2 in PAI_pocket:
                
                    PAI_concat= new_distance_mat[a1][a2]
                    if PAI_concat< 4.0:
                        PAI_concat_edges.append((a1,a2))
                        PAI_concat_fea.append([PAI_concat])
        assert len(PAI_concat_edges) == len(PAI_concat_fea)
        if len(PAI_concat_edges)==0:
           PAI_concat_edges.append((0,0))
           PAI_concat_fea.append([0.])
           PAI_cof.append([0])
           PAI_concat_fea = np.array(PAI_concat_fea,dtype=np.float32)
           PAI_cof = np.array(PAI_cof,dtype=np.float32)
        else:
            PAI_concat_fea = np.array(PAI_concat_fea,dtype=np.float32).reshape(-1,1)
            PAI_cof.append([1])
            PAI_cof = np.array(PAI_cof,dtype=np.float32)
            
        if PAI_concat_edges == [(0, 0)]:
            PAI_coe2 =np.zeros([num_atoms,1],dtype=np.float32)
        else:
            PAI_coe2 =np.ones([num_atoms,1],dtype=np.float32)
        assert PAI_coe2[0]== PAI_cof
        PAI_graph = pgl.Graph(PAI_concat_edges, num_nodes=num_atoms,node_feat={"coe": PAI_coe2} ,edge_feat={"dist": PAI_concat_fea, "coe": PAI_cof})
        return PAI_graph


    def making_van_graph(self, mol):
        VDW_radius={"6":1.70,"7":1.60,"8":1.55,"9":1.5,"11":2.4,"12":2.2,"15":1.95,"16":1.80,"17":1.8,"19":2.8,"20":2.40,"25":2.05,"26":2.05,"27":2.00,"28":2.00,"29":2.00,"30":2.1,"35":1.90,"38":2.55,"48":2.2,"53":2.10,"55":3.00}
        van_concat_edges=[]
        van_concat_fea=[]
        rep_att =[]
        num_atoms_d, coords, features, atoms, inter_feats, hbond_idx, hbond_feas, hbond_coe, hyb_ligand, hyb_pocket,name,NEW_DIST,NEW_EDGES, PAI_ligand, PAI_pocket= mol
        num_atoms = len(coords)
        dist_mat = distance.cdist(coords, coords, 'euclidean')
        np.fill_diagonal(dist_mat, np.inf)
        def lj_energy(r,  vdw_radius):
            return np.power(vdw_radius / r, 8)- 2*( np.power(vdw_radius / r, 4))
        dist_graph_base = dist_mat.copy()
        for a1 in range(num_atoms_d):
            for a2 in range(num_atoms_d,num_atoms):
                if dist_graph_base[a1][a2] !=0 and dist_graph_base[a1][a2] != np.inf:
                   van_concat_edges.append((a1,a2))
                   r1=VDW_radius[str(atoms[a1])]
                   r2=VDW_radius[str(atoms[a2])]
                   vdw_radius=r1+r2
                   vdw_force=lj_energy(dist_graph_base[a1][a2],vdw_radius)
                   if vdw_force>0:
                       print(vdw_force)
                   van_concat_fea.append([vdw_force])
                   rep_att.append(dist_graph_base[a1][a2])
                
        assert len(van_concat_edges) == len(van_concat_fea) == len(rep_att)
        van_concat_fea = np.array(van_concat_fea,dtype=np.float32).reshape(-1,1)
        rep_att= np.array(rep_att,dtype=np.float32).reshape(-1,1)
        van_graph = pgl.Graph(van_concat_edges, num_nodes=num_atoms, edge_feat={"dist": van_concat_fea,"rep_att":rep_att})
        return van_graph


    def load_data(self):
        """ Generate complex interaction graphs. """
        if self.has_cache():
            graphs, hbond_graphs,hyb_graphs,van_graphs,PAI_graphs,labels = self.load()
            self.a2a_graphs = graphs
            self.hbond_graph = hbond_graphs
            self.hyb_graph = hyb_graphs
            self.van_graph = van_graphs
            self.PAI_graph = PAI_graphs          
            self.labels = labels
        else:
            print('Processing raw protein-ligand complex data...')
            file_name = os.path.join(self.data_path, "{0}.pkl".format(self.dataset))
            with open(file_name, 'rb') as f:
                data_mols, data_Y = pickle.load(f)
            ALL=0
            for mol, y in tqdm(zip(data_mols, data_Y)):
                graphs,n = self.build_graph(mol)
                ALL+=n
                Hbond_graph = self.making_Hbond_graph(mol)
                hyb_graph = self.making_hyb_graph(mol)
                van_graph = self.making_van_graph(mol)
                PAI_graph= self.making_PAI_graph(mol)
                if graphs is None:
                    continue
                self.a2a_graphs.append(graphs)
                self.hbond_graph.append(Hbond_graph)
                self.hyb_graph.append(hyb_graph)
                self.van_graph.append(van_graph)
                self.PAI_graph.append(PAI_graph)
                self.labels.append(y)             
            self.labels = np.array(self.labels).reshape(-1, 1)
            if self.save_file:
                self.save()
            gc.collect()


def collate_fn(batch):
    a2a_gs,  Hbond_graphs, hyb_graphs,van_graphs, PAI_graphs,labels = map(list, zip(*batch))
    a2a_g = pgl.Graph.batch(a2a_gs).tensor()
    H_g = pgl.Graph.batch(Hbond_graphs).tensor()
    hyb_g = pgl.Graph.batch(hyb_graphs).tensor()
    van_g=pgl.Graph.batch(van_graphs).tensor()
    PAI_g=pgl.Graph.batch(PAI_graphs).tensor()
    labels = paddle.to_tensor(np.array(labels), dtype='float32')
    return a2a_g, H_g, hyb_g, van_g,PAI_g,labels
    gc.collect()


if __name__ == "__main__":
    complex_data = ComplexDataset("./data", "pdbbind2016_train", 2)
    loader = Dataloader(complex_data,
                        batch_size=1,
                        shuffle=False,
                        num_workers=1,
                        collate_fn=collate_fn)
    cc = 0
    for batch in loader:
        a2a_g,  Hbond_graphs, hyb_graphs, van_graphs,PAI_g,labels = batch
        cc += 1
        if cc == 1:
            break
