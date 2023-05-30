import torch
import torch.nn as nn
from memsearch.dataset import GRAPH_METADATA, REVERSED_GRAPH_METADATA
from torch_geometric.nn import GCNConv, HEATConv, HGTConv, HANConv, HeteroConv, SAGEConv
from torch_geometric.utils import add_self_loops
from pathlib import Path
from enum import Enum
import torch_geometric.transforms as T

class NodeFuseMethod(Enum):
    CONCAT = "concat"
    AVERAGE = "avg"
    MULTIPLY = "mult"

# TODO try TransformerConv, GATv2Conv,RGATConv, GENConv, GeneralConv
class MLPEdgeClassifier(torch.nn.Module):
    def __init__(self,
                 node_features_dim,
                 edge_features_dim,
                 node_embedding_dim = 32, 
                 edge_embedding_dim = 32, 
                 node_mlp_hidden_layers = [32, 32],
                 edge_mlp_hidden_layers = [32, 32],
                 output_mlp_hidden_layers = [32, 32],
                 node_fuse_method = NodeFuseMethod.CONCAT,
                 include_transformer=True,
                 reversed_edges=True):
        super().__init__()
        self.reversed_edges = reversed_edges
        self.node_embed_mlp = nn.ModuleList()
        fused_features_size = node_features_dim
        for layer_size in node_mlp_hidden_layers + [node_embedding_dim]:
            self.node_embed_mlp.append(nn.Linear(fused_features_size, layer_size))
            self.node_embed_mlp.append(nn.ReLU())
            fused_features_size = layer_size

        self.edge_embed_mlp = nn.ModuleList()
        fused_features_size = edge_features_dim
        for layer_size in edge_mlp_hidden_layers + [edge_embedding_dim]:
            self.edge_embed_mlp.append(nn.Linear(fused_features_size, layer_size))
            self.edge_embed_mlp.append(nn.ReLU())
            fused_features_size = layer_size

        self.node_fuse_method = node_fuse_method
        self.output_mlp = nn.ModuleList()
        if node_fuse_method != NodeFuseMethod.CONCAT:
            fused_features_size = node_embedding_dim + edge_embedding_dim
        else:
            fused_features_size = node_embedding_dim*2 + edge_embedding_dim

        self.include_transformer = include_transformer
        if self.include_transformer:
            transformer_layer = torch.nn.TransformerEncoderLayer(d_model=fused_features_size, 
                                                                 nhead=2, 
                                                                 dim_feedforward=fused_features_size, 
                                                                 dropout=0.25,
                                                                 batch_first=True)
            self.transformer_encoder = torch.nn.TransformerEncoder(transformer_layer, num_layers=2)

        for layer_size in output_mlp_hidden_layers:
            self.output_mlp.append(nn.Linear(fused_features_size, layer_size))
            self.output_mlp.append(nn.ReLU())
            fused_features_size = layer_size
        self.output_layer = nn.Linear(layer_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, 
                pyg_data,
                classify_edge_index,
                classify_edge_features):
        node_features = self.embed_nodes(pyg_data)
        edge_features = self.embed_edges(classify_edge_features)
        fused_features = self.fuse_features(node_features, edge_features, classify_edge_index)
        return self.classify_edges(fused_features)

    def embed_nodes(self, pyg_data):
        node_embeddings = pyg_data.x
        for layer in self.node_embed_mlp:
            node_embeddings = layer(node_embeddings)
        return node_embeddings

    def embed_edges(self, edge_features):
        edge_embeddings = edge_features
        for layer in self.edge_embed_mlp:
            edge_embeddings = layer(edge_embeddings)
        return edge_embeddings

    def fuse_features(self, node_embeddings, edge_embeddings, classify_edge_index):
        if self.node_fuse_method == NodeFuseMethod.CONCAT:
            fused_features = torch.cat([node_embeddings[classify_edge_index[:,:,0]],
                                        node_embeddings[classify_edge_index[:,:,1]]], dim=-1)
        elif self.node_fuse_method == NodeFuseMethod.AVERAGE:
            fused_features = (node_embeddings[classify_edge_index[:,:,0]] + 
                              node_embeddings[classify_edge_index[:,:,1]])/2.0
        elif self.node_fuse_method == NodeFuseMethod.MULTIPLY:
            fused_features = (node_embeddings[classify_edge_index[:,:,0]] * 
                              node_embeddings[classify_edge_index[:,:,1]])
        fused_features = torch.cat([fused_features, edge_embeddings], dim=-1)

        if self.include_transformer:
            padding_mask = (classify_edge_index[:,:,0].eq(0) & classify_edge_index[:,:,1].eq(0))
            fused_features[padding_mask] = 0.0
            fused_features = self.transformer_encoder(fused_features, src_key_padding_mask=padding_mask)

        return fused_features

    def classify_edges(self, fused_features):
        for layer in self.output_mlp:
            fused_features = layer(fused_features)
        output = self.sigmoid(self.output_layer(fused_features))
        return torch.squeeze(output)

    def is_heterogenous(self):
        return False

    def is_recurrent(self):
        return False

    def get_model_type(self):
        return type(self).__name__

class GNN(MLPEdgeClassifier):
    def __init__(self,
                 conv_layers,
                 node_features_dim,
                 edge_features_dim,
                 node_embedding_dim = 32, 
                 edge_embedding_dim = 32, 
                 node_mlp_hidden_layers = [32, 32],
                 edge_mlp_hidden_layers = [32, 32],
                 output_mlp_hidden_layers = [32, 32],
                 node_fuse_method = NodeFuseMethod.CONCAT,
                 include_transformer=True,
                 add_self_loops=True,
                 reversed_edges=True):
        super().__init__(
                 node_features_dim,
                 edge_features_dim,
                 node_embedding_dim, 
                 edge_embedding_dim, 
                 node_mlp_hidden_layers,
                 edge_mlp_hidden_layers,
                 output_mlp_hidden_layers,
                 node_fuse_method,
                 include_transformer,
                 reversed_edges)

        self.conv_layers = nn.ModuleList(conv_layers)
        self.add_self_loops = add_self_loops

    def forward(self, 
                pyg_data,
                classify_edge_index,
                classify_edge_features):
        if self.add_self_loops:
            pyg_data.edge_index, pyg_data.edge_attr = add_self_loops(pyg_data.edge_index, pyg_data.edge_attr)
        node_features = self.embed_nodes(pyg_data)
        edge_features = self.embed_edges(classify_edge_features)
        fused_features = self.fuse_features(node_features, edge_features, classify_edge_index)
        return self.classify_edges(fused_features)

    def embed_nodes(self, pyg_data):
        node_embeddings = pyg_data.x
        for layer in self.node_embed_mlp:
            node_embeddings = layer(node_embeddings)

        edge_index = pyg_data.edge_index
        # edge_weight = pyg_data.edge_attr
        for layer in self.conv_layers:
            node_embeddings = layer(node_embeddings, edge_index)
        return node_embeddings

class GCN(GNN):
    def __init__(self,
                 node_features_dim,
                 edge_features_dim,
                 node_embedding_dim = 32, 
                 edge_embedding_dim = 32, 
                 node_mlp_hidden_layers = [32, 32],
                 edge_mlp_hidden_layers = [32, 32],
                 gnn_hidden_layers=[32],
                 output_mlp_hidden_layers = [32, 32],
                 node_fuse_method = NodeFuseMethod.CONCAT,
                 include_transformer=True,
                 add_self_loops=True,
                 reversed_edges=True):
        conv_layers = []
        input_dim = node_embedding_dim
        for layer_size in gnn_hidden_layers+[node_embedding_dim]:
            conv_layers.append(GCNConv(input_dim, layer_size))
            input_dim = layer_size

        super().__init__(
                 conv_layers,
                 node_features_dim,
                 edge_features_dim,
                 node_embedding_dim, 
                 edge_embedding_dim, 
                 node_mlp_hidden_layers,
                 edge_mlp_hidden_layers,
                 output_mlp_hidden_layers,
                 node_fuse_method,
                 include_transformer,
                 add_self_loops,
                 reversed_edges)

class HEAT(GNN):
    def __init__(self,
                 node_features_dim,
                 edge_features_dim,
                 node_embedding_dim = 32, 
                 edge_embedding_dim = 32, 
                 node_mlp_hidden_layers = [32, 32],
                 edge_mlp_hidden_layers = [32, 32],
                 gnn_hidden_layers=[32],
                 output_mlp_hidden_layers = [32, 32],
                 node_fuse_method = NodeFuseMethod.CONCAT,
                 include_transformer=True,
                 add_self_loops=False,
                 reversed_edges=True,
                 num_node_types=5,
                 num_edge_types=5,
                 edge_type_emb_dim=2):
        conv_layers = []
        input_dim = node_embedding_dim
        for layer_size in gnn_hidden_layers+[node_embedding_dim]:
            conv = HEATConv(
                in_channels=input_dim,
                out_channels=layer_size,
                num_node_types=num_node_types,
                num_edge_types=num_edge_types,
                edge_type_emb_dim=edge_type_emb_dim,
                edge_dim=edge_embedding_dim,
                edge_attr_emb_dim=edge_embedding_dim,
                dropout=0.05,
                heads=1
            )
            conv_layers.append(conv)
            input_dim = layer_size
        super().__init__(
                 conv_layers,
                 node_features_dim,
                 edge_features_dim,
                 node_embedding_dim, 
                 edge_embedding_dim, 
                 node_mlp_hidden_layers,
                 edge_mlp_hidden_layers,
                 output_mlp_hidden_layers,
                 node_fuse_method,
                 include_transformer,
                 add_self_loops,
                 reversed_edges)

    def forward(self, 
                pyg_data,
                classify_edge_index,
                classify_edge_features):
        if self.add_self_loops:
            pyg_data.edge_index, pyg_data.edge_attr = add_self_loops(pyg_data.edge_index, pyg_data.edge_attr)
        node_types, edge_types, classify_edge_features = self.extract_type_lists(pyg_data, classify_edge_features)
        node_features = self.embed_nodes(pyg_data, node_types, edge_types)
        edge_features = self.embed_edges(classify_edge_features)
        fused_features = self.fuse_features(node_features, edge_features, classify_edge_index)
        return self.classify_edges(fused_features)

    def embed_nodes(self, pyg_data, node_types, edge_types):
        edge_index = pyg_data.edge_index
        edge_attr = self.embed_edges(pyg_data.edge_attr)

        edge_types = edge_types.cpu().tolist() + [4]*(edge_index.size(1)-len(edge_types))
        edge_types = torch.LongTensor(edge_types).cuda()

        node_embeddings = pyg_data.x
        for layer in self.node_embed_mlp:
            node_embeddings = layer(node_embeddings)

        for layer in self.conv_layers:
            # Edge types and edge_index don't line up
            node_embeddings = layer(node_embeddings, edge_index, node_types, edge_types, edge_attr).relu()
        return node_embeddings

    @staticmethod
    def extract_type_lists(data, loss_edge_features):
        # a bunch of slighly hacky stuff specificaly for HEAT model
        # it requires a list of node and edge types instead of a dict like other heterog pyg models
        # so we extract those from the node/edge feature vectors, and remove those dims from the feature vecs
        node_attr = data.x.detach().cpu().numpy()
        edge_attr = data.edge_attr.detach().cpu().numpy()
        # -5 because the five last spots in vector are the type 1-hot code
        edge_types = torch.LongTensor(edge_attr[:,-5:].nonzero()[1])
        node_types = torch.LongTensor(node_attr[:,-5:].nonzero()[1])
        x = torch.Tensor(node_attr[:,:-5])
        edge_attr = torch.Tensor(edge_attr[:,:-5]).cuda()
        node_types = node_types.cuda()
        edge_types = edge_types.cuda()
        data.x = x.cuda()
        data.edge_attr = edge_attr.cuda()
        loss_edge_features = loss_edge_features.detach().cpu().numpy()
        loss_edge_features = torch.Tensor(loss_edge_features[:,:,:-5]).cuda()
        return node_types, edge_types, loss_edge_features

class HGNN(GNN):

    def is_heterogenous(self):
        return True
    #
    def forward(self, 
                pyg_data,
                classify_edge_index,
                classify_edge_features,
                edge_key):
        if self.add_self_loops:
            pyg_data = T.AddSelfLoops()(pyg_data)

        node_features = self.embed_nodes(pyg_data)
        edge_features = self.embed_edges(classify_edge_features)
        fused_features = self.fuse_features(node_features, edge_features, classify_edge_index, edge_key)
        return self.classify_edges(fused_features)

    def embed_nodes(self, pyg_data):
        node_embeddings = pyg_data.x_dict
        for layer in self.node_embed_mlp:
            node_embeddings = {x: layer(node_embeddings[x]) for x in node_embeddings}

        for conv in self.conv_layers:
            node_embeddings = conv(node_embeddings, pyg_data.edge_index_dict)
            node_embeddings = {x: y.relu() for x, y in node_embeddings.items()}

        return node_embeddings

    def embed_edges(self, edge_features):
        edge_embeddings = edge_features
        for layer in self.edge_embed_mlp:
            edge_embeddings = layer(edge_embeddings)
        return edge_embeddings

    def fuse_features(self, node_embeddings, edge_embeddings, classify_edge_index, classify_edge_key):

        if self.node_fuse_method == NodeFuseMethod.CONCAT:
            fused_features = torch.cat([node_embeddings[classify_edge_key[0]][classify_edge_index[:,:,0]],
                                        node_embeddings[classify_edge_key[2]][classify_edge_index[:,:,1]]], dim=-1)
        elif self.node_fuse_method == NodeFuseMethod.AVERAGE:
            fused_features = (node_embeddings[classify_edge_key[0]][classify_edge_index[:,:,0]] + 
                              node_embeddings[classify_edge_key[2]][classify_edge_index[:,:,1]])/2.0
        elif self.node_fuse_method == NodeFuseMethod.MULTIPLY:
            fused_features = (node_embeddings[classify_edge_key[0]][classify_edge_index[:,:,0]] * 
                              node_embeddings[classify_edge_key[2]][classify_edge_index[:,:,1]])

        fused_features = torch.cat([fused_features, edge_embeddings], dim=-1)

        if self.include_transformer:
            padding_mask = (classify_edge_index[:,:,0].eq(0) & classify_edge_index[:,:,1].eq(0))
            fused_features[padding_mask] = 0.0
            fused_features = self.transformer_encoder(fused_features, src_key_padding_mask=padding_mask)

        return fused_features

class HGCN(HGNN):
    def __init__(self,
                 node_features_dim,
                 edge_features_dim,
                 node_embedding_dim = 32, 
                 edge_embedding_dim = 32, 
                 node_mlp_hidden_layers = [32, 32],
                 edge_mlp_hidden_layers = [32, 32],
                 gnn_hidden_layers=[32],
                 output_mlp_hidden_layers = [32, 32],
                 node_fuse_method = NodeFuseMethod.CONCAT,
                 include_transformer=True,
                 add_self_loops=True,
                 reversed_edges=True):
        conv_layers = []
        metadata = REVERSED_GRAPH_METADATA if reversed_edges else GRAPH_METADATA
        input_dim = node_embedding_dim
        for layer_size in gnn_hidden_layers+[node_embedding_dim]:
            for key in metadata['edge_types']:
                conv_layers_dict = {}
                if key [0] == key[2]:
                    conv_layers_dict[key] = GCNConv(input_dim, node_embedding_dim)
                    conv_layers_dict[key] = GCNConv(input_dim, node_embedding_dim)
                else:
                    conv_layers_dict[key] = SAGEConv(input_dim, node_embedding_dim)
                    conv_layers_dict[key] = SAGEConv(input_dim, node_embedding_dim)
                conv_layers.append(HeteroConv(conv_layers_dict, aggr='mean'))
                input_dim = layer_size

        super().__init__(
                 conv_layers,
                 node_features_dim,
                 edge_features_dim,
                 node_embedding_dim, 
                 edge_embedding_dim, 
                 node_mlp_hidden_layers,
                 edge_mlp_hidden_layers,
                 output_mlp_hidden_layers,
                 node_fuse_method,
                 include_transformer,
                 add_self_loops,
                 reversed_edges)

class HGT(HGNN):
    def __init__(self,
                 node_features_dim,
                 edge_features_dim,
                 node_embedding_dim = 32, 
                 edge_embedding_dim = 32, 
                 node_mlp_hidden_layers = [32, 32],
                 gnn_hidden_layers=[32],
                 edge_mlp_hidden_layers = [32, 32],
                 output_mlp_hidden_layers = [32, 32],
                 node_fuse_method = NodeFuseMethod.CONCAT,
                 include_transformer=True,
                 add_self_loops=True,
                 reversed_edges=True):
        metadata = REVERSED_GRAPH_METADATA if reversed_edges else GRAPH_METADATA
        metadata = (metadata['node_types'], metadata['edge_types'])
        conv_layers = []
        input_dim = node_embedding_dim
        for layer_size in gnn_hidden_layers+[node_embedding_dim]:
            conv_layers.append(HGTConv(input_dim, node_embedding_dim, metadata))
            input_dim = layer_size

        super().__init__(
                 conv_layers,
                 node_features_dim,
                 edge_features_dim,
                 node_embedding_dim, 
                 edge_embedding_dim, 
                 node_mlp_hidden_layers,
                 edge_mlp_hidden_layers,
                 output_mlp_hidden_layers,
                 node_fuse_method,
                 include_transformer,
                 add_self_loops,
                 reversed_edges)

class HAN(HGNN):
    def __init__(self,
                 node_features_dim,
                 edge_features_dim,
                 node_embedding_dim = 32, 
                 edge_embedding_dim = 32, 
                 node_mlp_hidden_layers = [32, 32],
                 edge_mlp_hidden_layers = [32, 32],
                 gnn_hidden_layers=[32],
                 output_mlp_hidden_layers = [32, 32],
                 node_fuse_method = NodeFuseMethod.CONCAT,
                 include_transformer=True,
                 add_self_loops=True,
                 reversed_edges=True):
        metadata = REVERSED_GRAPH_METADATA if reversed_edges else GRAPH_METADATA
        metadata = (metadata['node_types'], metadata['edge_types'])
        conv_layers = []
        input_dim = node_embedding_dim
        for layer_size in gnn_hidden_layers+[node_embedding_dim]:
            conv_layers.append(HANConv(input_dim, node_embedding_dim, metadata))
            input_dim = layer_size

        super().__init__(
                 conv_layers,
                 node_features_dim,
                 edge_features_dim,
                 node_embedding_dim, 
                 edge_embedding_dim, 
                 node_mlp_hidden_layers,
                 edge_mlp_hidden_layers,
                 output_mlp_hidden_layers,
                 node_fuse_method,
                 include_transformer,
                 add_self_loops,
                 reversed_edges)

def make_model(cfg, node_featurizer, edge_featurizer, load_model=False):
    node_features_dim = node_featurizer.get_feature_size()
    edge_features_dim = edge_featurizer.get_feature_size()
    if cfg.add_num_nodes:
        edge_features_dim+=1
    if cfg.add_num_edges:
        edge_features_dim+=1
    if cfg.model_type == 'heat':
        node_features_dim = node_features_dim-5
        edge_features_dim = edge_features_dim-5

    assert cfg.node_fuse_method in [x.value for x in NodeFuseMethod]
    node_fuse_method = [x for x in NodeFuseMethod if x.value==cfg.node_fuse_method][0]
    args = {'node_features_dim':node_features_dim, 
            'edge_features_dim':edge_features_dim,
            'node_embedding_dim':cfg.node_embedding_dim,
            'edge_embedding_dim':cfg.edge_embedding_dim,
            'node_mlp_hidden_layers':cfg.node_mlp_hidden_layers,
            'edge_mlp_hidden_layers':cfg.edge_mlp_hidden_layers,
            'output_mlp_hidden_layers':cfg.output_mlp_hidden_layers,
            'node_fuse_method':node_fuse_method,
            'include_transformer':cfg.include_transformer,
            'reversed_edges':cfg.reversed_edges}

    if 'add_self_loops' in cfg:
        cfg['add_self_loops'] =  cfg.add_self_loops
    if cfg.model_type != 'mlp':
        args['gnn_hidden_layers'] = cfg.gnn_hidden_layers
    model = None
    if cfg.model_type == 'mlp':
        model = MLPEdgeClassifier(**args)
    elif cfg.model_type == 'gcn':
        model = GCN(**args)
    elif cfg.model_type == 'heat':
        args['edge_type_emb_dim'] = cfg.edge_type_emb_dim
        args['num_node_types'] = 5
        args['num_edge_types'] = 5
        model = HEAT(**args)
    elif cfg.model_type == 'hgt':
        model = HGT(**args)
    elif cfg.model_type == 'hgcn':
        model = HGCN(**args)
    elif cfg.model_type == 'han':
        model = HAN(**args)
    else:
        Exception(f"Invalid model type: {cfg.model_type}")

    if load_model:
        model_path = create_path_to_model(cfg, node_featurizer, edge_featurizer)
        model.load_state_dict(torch.load(model_path))
    model.cuda()

    return model

def create_path_to_model(cfg, node_featurizer, edge_featurizer):
    Path(cfg.models_dir).mkdir(exist_ok=True)
    model_path = '%s/%s.pt'%(cfg.models_dir,
                             cfg.model_name)
    return model_path

def compute_output(model, input_data, edges, edge_features, edge_key=None):
    if model.is_heterogenous():
        out = model(input_data, edges, edge_features, edge_key)
    else:
        out = model(input_data, edges, edge_features)
    return out

'''
class RecurrentGCNNet(torch.nn.Module):
    def __init__(self, node_features_dim=96, hidden_dim=32, gcn_out_dim=16, use_classifier_mlp=True):
        super().__init__()
        self.conv1 = GCLSTM(node_features_dim, hidden_dim, 4)
        self.conv2 = GCNConv(hidden_dim, gcn_out_dim)
        self.use_classifier_mlp = use_classifier_mlp
        if self.use_classifier_mlp:
            self.mlp = nn.Linear(gcn_out_dim*2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, classify_edge_index, h, c):
        h, c = self.conv1(x, edge_index, H=h, C=c)
        x = h.relu()
        node_features = self.conv2(x, edge_index)
        if self.use_classifier_mlp:
            node_pair_features = torch.cat([node_features[classify_edge_index[0]],
                                            node_features[classify_edge_index[1]]], dim=-1)
            mlp_output = torch.squeeze(self.mlp(node_pair_features))
            out = self.sigmoid(mlp_output)
        else:
            out = self.sigmoid((node_features[classify_edge_index[0]] * \
                                node_features[classify_edge_index[1]]).sum(dim=-1)).view(-1)
        return out, h, c

def compute_output_recurrent(model, input_data, edges, edge_features, h, c):
    return model(input_data.x, input_data.edge_index, edges, edge_features, h, c)
'''
