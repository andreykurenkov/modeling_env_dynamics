import os
import torch
import networkx
import spacy
import functools
import numpy as np
import pickle 

from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from memsearch.graphs import NodeType, EdgeType, RECIPROCAL_EDGE_TYPES
from collections import defaultdict
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

POSSIBLE_NODE_FEATURES = ['text_embedding',
                          'time_since_observed',
                          'times_observed',
                          'time_since_state_change',
                          'times_state_changed',
                          'state_change_freq',
                          'node_type']
POSSIBLE_EDGE_FEATURES = ['cosine_similarity',
                          'time_since_observed',
                          'time_since_state_change',
                          'times_observed',
                          'times_state_true',
                          'freq_true',
                          'times_state_changed',
                          'state_change_freq',
                          'last_observed_state',
                          'prior_prob',
                          'edge_type']

GRAPH_METADATA = {
        'node_types': ['house','floor','room','furniture','object'], 
        'edge_types': [
         ('floor', 'in', 'house'),
         ('house', 'contains', 'floor'),
         ('floor', 'contains', 'room'),
         ('room', 'in', 'floor'),
         ('furniture', 'in', 'room'), 
         ('room', 'contains', 'furniture'),
         ('object', 'in', 'furniture'),
         ('furniture', 'contains', 'object'),
         ('object', 'onTop', 'furniture'),
         ('furniture', 'under', 'object'),
        ]}
        
REVERSED_GRAPH_METADATA = {
        'node_types': ['house','floor','room','furniture','object'], 
        'edge_types': [(key[2], RECIPROCAL_EDGE_TYPES[EdgeType(key[1])].value, key[0]) for key in GRAPH_METADATA['edge_types']]
        }

def save_networkx_graph(save_path, graph, node_featurizer, edge_featurizer):
    folder_path = os.path.dirname(save_path)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    networkx_graph = graph_to_networkx(graph, node_featurizer, edge_featurizer)
    networkx.write_gpickle(networkx_graph,save_path)

def node_type_to_one_hot(node_type):
    # this is faster than for loop
    if node_type == NodeType.HOUSE:
        return [1,0,0,0,0]
    elif node_type == NodeType.FLOOR:
        return [0,1,0,0,0]
    elif node_type == NodeType.ROOM:
        return [0,0,1,0,0]
    elif node_type == NodeType.FURNITURE:
        return [0,0,0,1,0]
    elif node_type == NodeType.OBJECT:
        return [0,0,0,0,1]

def edge_type_to_one_hot(edge_type):
    if edge_type == EdgeType.ONTOP:
        return [1,0,0,0,0]
    elif edge_type == EdgeType.IN:
        return [0,1,0,0,0]
    elif edge_type == EdgeType.CONTAINS:
        return [0,0,1,0,0]
    elif edge_type == EdgeType.UNDER:
        return [0,0,0,1,0]
    elif edge_type == EdgeType.CONNECTED:
        return [0,0,0,0,1]

def graph_to_networkx(graph, node_featurizer = None, edge_featurizer = None):
    """
    Convert from Graph in graphs.py to networkx graph
    """
    G = networkx.MultiDiGraph()
    for node in graph.nodes:
        G.add_node(node.unique_id)
        if node_featurizer is None:
            continue
        node_features = node_featurizer.featurize(node, add_type=True)
        for feature_name in node_features:
            G.nodes[node.unique_id][feature_name] = node_features[feature_name]

    for edge in graph.get_edges():
        u = edge.node1.unique_id
        v = edge.node2.unique_id
        edge_type = edge.type.value
        if edge.prob is not None:
            G.add_edge(u, v, key=edge_type, weight=edge.prob)
        else:
            G.add_edge(u, v, key=edge_type)
        if edge_featurizer is None:
            continue
        edge_features = edge_featurizer.featurize(edge, add_type=True)
        edge_features['should_sample_for_loss'] = edge.is_query_edge
        for feature_name in edge_features:
            G.edges[u, v, edge_type][feature_name] = edge_features[feature_name]
    return G

def graph_to_pyg(graph, node_featurizer = None, edge_featurizer = None, using_networkx = True, heterogenous=False, include_labels=True, graph_metadata={}, reverse_edges=True, add_mapping=False):
    """
    Convert from Graph in graphs.py to pytorch geometric graph
    """
    if using_networkx:
        networkx_graph = graph_to_networkx(graph, node_featurizer, edge_featurizer)
        pyg_graph = networkx_to_pyg(networkx_graph, node_featurizer, edge_featurizer, 
                heterogenous=heterogenous, include_labels=include_labels, 
                graph_metadata=graph_metadata, reverse_edges=reverse_edges, add_mapping=add_mapping)
        return pyg_graph

    node_matrix = []
    for node in graph.nodes:
        node_matrix.append(node_featurizer.featurize_to_vec(node))

    edges = []
    edge_weights = []
    for edge in graph.get_edges():
        edges.append(np.array([graph.get_node_num(edge.node1),
                               graph.get_node_num(edge.node2)]))
        edge_weights.append(np.array([edge.prob]))

    pyg_x = torch.tensor(np.stack(node_matrix), dtype=torch.float).cuda()
    pyg_edge_index = torch.tensor(np.stack(edges, axis=-1), dtype=torch.long).cuda()
    pyg_edge_weights = torch.tensor(np.concatenate(edge_weights), dtype=torch.float).cuda()
    return pyg_x, pyg_edge_index, pyg_edge_weights

def networkx_to_pyg_homogeneous(graph, node_featurizer, edge_featurizer, include_labels, reverse_edges=True): 
    data = Data()
    node_list = []

    # nodes
    node_nums = {}
    node_num = 0
    for node_id, node_features in graph.nodes(data=True):
        node_nums[node_id] = node_num
        node_num+=1
        features_vec = combine_features_to_vec(node_features, node_featurizer.features)
        node_list.append(features_vec)
    data.x = torch.Tensor(np.array(node_list))

    edge_list = [[],[]]
    edge_feature_vecs_list = []
    edge_label_list = []
    edge_should_sample_for_loss_list = []
    for source_node, target_node, edge_features in graph.edges(data=True):
        features_vec = combine_features_to_vec(edge_features, edge_featurizer.features)
        source_node_num = node_nums[source_node]
        target_node_num = node_nums[target_node]
        if reverse_edges:
            edge_list[0].append(target_node_num)
            edge_list[1].append(source_node_num)
        else:
            edge_list[0].append(source_node_num)
            edge_list[1].append(target_node_num)
        edge_feature_vecs_list.append(features_vec)
        if include_labels:
            edge_label_list.append(edge_features['label'])
            edge_should_sample_for_loss_list.append(int(edge_features['should_sample_for_loss']))

    data.edge_index = torch.LongTensor(np.array(edge_list))
    data.edge_attr = torch.Tensor(np.array(edge_feature_vecs_list))
    if include_labels:
        data.y = torch.Tensor(np.array(edge_label_list))
        data.should_sample_for_loss = torch.Tensor(np.array(edge_should_sample_for_loss_list))
    return data

def networkx_to_pyg_heterogeneous(graph, node_featurizer, edge_featurizer, include_labels, graph_metadata, reverse_edges=True, add_mapping=False): 
    data = HeteroData()

    node_lists = {'house': [], 'room':[], 'floor': [], 'furniture':[], 'object': []}
    node_id_lists = {'house': [], 'room':[], 'floor': [], 'furniture':[], 'object': []}

    # nodes
    node_nums = defaultdict(dict)
    counts_by_type = {node_type : 0 for node_type in node_lists}
    for node_id, node_features in graph.nodes(data=True):
        node_type = node_features['type']
        node_nums[node_type][node_id] = counts_by_type[node_type]
        node_id_lists[node_type].append(node_id)
        counts_by_type[node_type] += 1
        features_vec = combine_features_to_vec(node_features, node_featurizer.features)
        node_lists[node_type].append(features_vec)

    for node_type in node_lists:
        data[node_type].x = torch.Tensor(np.array(node_lists[node_type]))
        if add_mapping == True:
            data[node_type].mapping = node_nums[node_type]
            data[node_type].id = np.array(node_id_lists[node_type])

    edge_lists = defaultdict(lambda: [[],[]])
    edge_feature_vecs_lists = defaultdict(lambda: [])
    edge_label_lists = defaultdict(lambda: [])
    edge_should_sample_for_loss_lists = defaultdict(lambda: [])

    for source_node, target_node, edge_features in graph.edges(data=True):
        source_node_type = graph.nodes[source_node]['type']
        target_node_type = graph.nodes[target_node]['type']
        source_node_num = node_nums[source_node_type][source_node]
        target_node_num = node_nums[target_node_type][target_node]

        assert source_node_num < counts_by_type[source_node_type]
        assert target_node_num < counts_by_type[target_node_type]
        assert source_node_num < data[source_node_type].x.shape[0]
        assert target_node_num < data[target_node_type].x.shape[0]

        edge_type = edge_features['type']
        features_vec = combine_features_to_vec(edge_features, edge_featurizer.features)

        if reverse_edges:
            key = (target_node_type, RECIPROCAL_EDGE_TYPES[EdgeType(edge_type)].value, source_node_type)
        else:
            key = (source_node_type, edge_type, target_node_type)

        edge_list = edge_lists[key]
        if reverse_edges:
            edge_list[0].append(target_node_num)
            edge_list[1].append(source_node_num)
        else:
            edge_list[0].append(source_node_num)
            edge_list[1].append(target_node_num)

        edge_feature_vecs_list = edge_feature_vecs_lists[key]
        edge_feature_vecs_list.append(features_vec)

        if include_labels:
            edge_label_list = edge_label_lists[key]
            edge_should_sample_for_loss_list = edge_should_sample_for_loss_lists[key]
            edge_label_list.append(edge_features['label'])
            edge_should_sample_for_loss_list.append(int(edge_features['should_sample_for_loss']))

    # Ensure all graphs have homogenous keys/edge types
    for key in edge_lists:
        data[key].edge_index = torch.LongTensor(np.array(edge_lists[key]))
        data[key].edge_attr = torch.Tensor(np.array(edge_feature_vecs_lists[key]))
        if include_labels:
            data[key].y = torch.Tensor(np.array(edge_label_lists[key]))
            data[key].should_sample_for_loss = torch.Tensor(np.array(edge_should_sample_for_loss_lists[key]))

    for edge_type in graph_metadata["edge_types"]:
        if edge_type not in data.edge_types:
            data[edge_type].edge_index = torch.empty(2, 0, dtype=torch.long)
            data[edge_type].edge_attr = torch.empty(0, edge_featurizer.get_feature_size(), dtype=torch.float32)
            if include_labels:
                data[edge_type].y = torch.empty(0, dtype=torch.float32)
                data[edge_type].should_sample_for_loss = torch.empty(0, dtype=torch.float32)

    for node_type in graph_metadata["node_types"]:
        if node_type not in data.node_types:
            data[node_type] = torch.empty(0, node_featurizer.get_feature_size(), dtype=torch.float32)

    for key, value in data.edge_index_dict.items():
        if value.shape[1] != 0:
            assert value[0, :].max() <= data.x_dict[key[0]].shape[0]  - 1
            assert value[1, :].max() <= data.x_dict[key[2]].shape[0]  - 1

    # for key, value in data.edge_index_dict.items():
    #     if value.shape[1] != 0:
    #         print("Node from", key[0], ": ", data.x_dict[key[0]].shape[0])
    #         print("Node to", key[2], ": ", data.x_dict[key[2]].shape[0])
    #         print("Edge from ", key[0], ": ", value[0, :].max())
    #         print("Edge to ", key[2], ": ", value[1, :].max())

    return data

def networkx_to_pyg(graph, node_featurizer, edge_featurizer, include_labels, heterogenous=False, graph_metadata={}, reverse_edges=True, add_mapping=False):
    if heterogenous:
        return networkx_to_pyg_heterogeneous(graph, node_featurizer, edge_featurizer, include_labels, graph_metadata, reverse_edges, add_mapping=add_mapping)
    else:
        return networkx_to_pyg_homogeneous(graph, node_featurizer, edge_featurizer, include_labels, reverse_edges)

def networkx_file_to_pyg(file_name, node_featurizer, edge_featurizer, include_labels, heterogenous=False, graph_metadata=None, reverse_edges=True):
    return networkx_to_pyg(pickle.load(file_name), node_featurizer, edge_featurizer, include_labels, heterogenous, graph_metadata, reverse_edges)

def combine_features_to_vec(features_dict, features):
    features_list = []
    for feature_name in features:
        feature = features_dict[feature_name]
        if type(feature) is float or type(feature) is int:
            features_list.append(feature)
        else:
            features_list+=list(feature)
    return features_list

class NodeFeaturizer(object):

    def __init__(self, features, steps_per_scene=250, scene_avg_num_nodes=200, embed_text_with_transformer=False):
        self.features_size = 0
        self.steps_per_scene = float(steps_per_scene)
        self.scene_avg_num_nodes = float(scene_avg_num_nodes)
        self.temporal_feature_multiplier = scene_avg_num_nodes/10.0 #hacky way to normalize features
        if features=='all':
            features = POSSIBLE_NODE_FEATURES
        self.text_embeddings_cache = self.load_text_embedding_dict()
        self.features = features
        self.embed_text_with_transformer = embed_text_with_transformer
        if embed_text_with_transformer:
            self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.text_embedder = spacy.load("en_core_web_sm")
        if 'text_embedding' in features:
            if embed_text_with_transformer:
                self.features_size+=384
            else:
                self.features_size+=96
        if 'time_since_observed' in features:
            self.features_size+=1
        if 'times_observed' in features:
            self.features_size+=1
        if 'time_since_state_change' in features:
            self.features_size+=1
        if 'times_state_changed' in features:
            self.features_size+=1
        if 'state_change_freq' in features:
            self.features_size+=1
        if 'node_type' in features:
            self.features_size+=5

    def load_text_embedding_dict(self):
        if os.path.isfile('outputs/text_embed_cache.pkl'):
            with open('outputs/text_embed_cache.pkl','rb') as f:
                return pickle.load(f)
        else:
            return {}

    def save_text_embedding_dict(self):
        with open('outputs/text_embed_cache.pkl','wb') as f:
            return pickle.dump(self.text_embeddings_cache, f)

    def get_node_text_embedding(self, node):
        node_description = node.label#' '.join(node.description.split('_'))
        if node_description in self.text_embeddings_cache:
            embedding = self.text_embeddings_cache[node_description]
        else:
            if self.embed_text_with_transformer:
                embedding = self.text_embedder.encode(node_description, show_progress_bar=False)
            else:
                word_vecs = []
                for word in node_description.split(' '):
                    if word in self.text_embeddings_cache:
                        word_vecs.append(self.text_embeddings_cache[word])
                    else:
                        word_vec = self.text_embedder(word).vector
                        word_vecs.append(word_vec)
                        self.text_embeddings_cache[word] = word_vec
                embedding = np.mean(word_vecs, axis=0)
            self.text_embeddings_cache[node_description] = embedding
        return embedding

    def featurize(self, node, add_type = False):
        node_features = {}
        if 'text_embedding' in self.features:
            node_features['text_embedding'] = self.get_node_text_embedding(node)
        if 'time_since_observed' in self.features:
            node_features['time_since_observed'] = float(node.time_since_observed) / self.steps_per_scene * self.temporal_feature_multiplier
        if 'times_observed' in self.features:
            node_features['times_observed'] = float(node.times_observed)/self.steps_per_scene * self.temporal_feature_multiplier
        if 'time_since_state_change' in self.features:
            node_features['time_since_state_change'] = float(node.time_since_state_change)/self.steps_per_scene * self.temporal_feature_multiplier
        if 'times_state_changed' in self.features:
            node_features['times_state_changed'] = float(node.times_state_changed)/self.steps_per_scene * self.temporal_feature_multiplier
        if 'state_change_freq' in self.features:
            node_features['state_change_freq'] = float(node.state_change_freq)/self.steps_per_scene * self.temporal_feature_multiplier
        if 'node_type' in self.features:
            node_features['node_type'] = node_type_to_one_hot(node.type)
        for feature in self.features:
            assert feature in node_features
        if add_type:
            node_features['type'] = node.type.value
            
        return node_features

    def featurize_to_vec(self, node):
        features_dict = self.featurize(node)
        return combine_features_to_vec(features_dict, self.features)

    def get_feature_size(self):
        return self.features_size

    def get_numeric_code(self):
        node_features_str = ''
        for node_feature in self.features:
            node_features_str+=str(POSSIBLE_NODE_FEATURES.index(node_feature))
        return node_features_str

class EdgeFeaturizer(object):

    def __init__(self, features, node_featurizer=None, include_labels=False, zero_priors_to_objects=False, steps_per_scene=250, scene_avg_num_edges=500):
        self.features_size = 0
        self.steps_per_scene = steps_per_scene
        self.scene_avg_num_edges = scene_avg_num_edges
        self.temporal_feature_multiplier = scene_avg_num_edges/10.0 #hacky way to normalize features
        self.node_featurizer = node_featurizer
        if features=='all':
            features = POSSIBLE_EDGE_FEATURES
        self.cos_similarity_cache = {}
        self.include_labels = include_labels
        self.zero_priobs_to_objects = zero_priors_to_objects
        self.features = features
        if 'cosine_similarity' in features:
            self.features_size+=1
        if 'time_since_observed' in features:
            self.features_size+=1
        if 'time_since_state_change' in features:
            self.features_size+=1
        if 'times_observed' in features:
            self.features_size+=1
        if 'times_state_true' in features:
            self.features_size+=1
        if 'times_state_changed' in features:
            self.features_size+=1
        if 'state_change_freq' in features:
            self.features_size+=1
        if 'last_observed_state' in features:
            self.features_size+=1
        if 'freq_true' in features:
            self.features_size+=1
        if 'edge_type' in features:
            self.features_size+=5
        if 'prior_prob' in features:
            self.features_size+=1

    def featurize(self, edge, add_type=False):
        edge_features = {}
        if 'cosine_similarity' in self.features:
            emb1 = self.node_featurizer.get_node_text_embedding(edge.node1)
            emb2 = self.node_featurizer.get_node_text_embedding(edge.node2)
            dict_key = (edge.node1.label, edge.node2.label)
            if dict_key in self.cos_similarity_cache:
                edge_features['cosine_similarity'] = self.cos_similarity_cache[dict_key]
            else:
                cos_sim = dot(emb1, emb2)/(norm(emb1)*norm(emb2))
                edge_features['cosine_similarity'] = float(cos_sim)
                self.cos_similarity_cache[dict_key] = float(cos_sim)
        if 'time_since_observed' in self.features:
            edge_features['time_since_observed'] = edge.time_since_observed/self.steps_per_scene 
        if 'time_since_state_change' in self.features:
            edge_features['time_since_state_change'] = edge.time_since_state_change/self.steps_per_scene * self.temporal_feature_multiplier
        if 'times_state_changed' in self.features:
            edge_features['times_state_changed'] = float(edge.times_state_changed)/self.steps_per_scene * self.temporal_feature_multiplier
        if 'times_observed' in self.features:
            edge_features['times_observed'] = edge.time_since_state_change/self.steps_per_scene * self.temporal_feature_multiplier
        if 'times_state_true' in self.features:
            edge_features['times_state_true'] = float(edge.times_state_true)/self.steps_per_scene * self.temporal_feature_multiplier
        if 'times_observed' in self.features:
            edge_features['times_observed'] = float(edge.times_observed)/self.steps_per_scene * self.temporal_feature_multiplier
        if 'state_change_freq' in self.features:
            edge_features['state_change_freq'] = edge.state_change_freq/self.steps_per_scene * self.temporal_feature_multiplier
        if 'last_observed_state' in self.features:
            edge_features['last_observed_state'] = int(edge.last_observed_state)
        if 'freq_true' in self.features:
            edge_features['freq_true'] = edge.freq_true
        if 'edge_type' in self.features:
            edge_features['edge_type'] = edge_type_to_one_hot(edge.type)
        if 'prior_prob' in self.features:
            if edge.node2.type != NodeType.OBJECT:
                edge_features['prior_prob'] = 1.0
            elif self.zero_priobs_to_objects:
                edge_features['prior_prob'] = 0.0
            else:
                edge_features['prior_prob'] = edge.prob
        if self.include_labels:
            edge_features['label'] = int(edge.currently_true)
        for feature in self.features:
            assert feature in edge_features
        if add_type:
            edge_features['type'] = edge.type.value
        return edge_features

    def featurize_hypothetical_edge(self, edge_type):
        edge_features = {}
        if 'time_since_observed' in self.features:
            edge_features['time_since_observed'] = -1
        if 'times_observed' in self.features:
            edge_features['times_observed'] = 0
        if 'edge_type' in self.features:
            edge_features['edge_type'] = edge_type_to_one_hot(edge_type)
        return edge_features

    def featurize_to_vec(self, edge):
        features_dict = self.featurize(edge)
        return combine_features_to_vec(features_dict, self.features)

    def get_feature_size(self):
        return self.features_size

    def get_numeric_code(self):
        edge_features_str = ''
        for edge_feature in self.features:
            edge_features_str+=str(POSSIBLE_EDGE_FEATURES.index(edge_feature))
        return edge_features_str

class SGMDataset(InMemoryDataset):

    def __init__(self,
                 input_files_dir,
                 output_files_dir,
                 node_featurizer,
                 edge_featurizer,
                 transform=None,
                 include_labels=False,
                 add_num_nodes=False,
                 add_num_edges=False,
                 pre_transform=None,
                 pre_filter=None,
                 num_workers=1,
                 heterogenous=False,
                 parse_metadata=False,
                 reverse_edges=True,
                 data=None):
        self.input_files_dir = input_files_dir
        self.output_files_dir = output_files_dir
        self.node_featurizer = node_featurizer
        self.edge_featurizer = edge_featurizer
        self.node_features = node_featurizer.features
        self.edge_features = edge_featurizer.features
        self.include_labels = include_labels
        self.add_num_nodes = add_num_nodes
        self.add_num_edges = add_num_edges
        self.num_workers = num_workers
        self.heterogenous = heterogenous
        self.parse_metadata = parse_metadata
        self.reverse_edges = reverse_edges
        self.data = data
        #not using root in super classes
        super().__init__(None, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        paths = os.listdir(self.input_files_dir)
        # Do consistent sorting of paths from start to finish
        paths.sort()
        return paths

    @property
    def processed_file_names(self):
        if self.heterogenous:
            return ['heterogenous_dataset.pt']
        else:
            return ['homogenous_dataset.pt']

    @property
    def raw_dir(self):
        return self.input_files_dir

    @property
    def processed_dir(self):
        return self.output_files_dir

    @property
    def raw_dir(self):
        return self.input_files_dir

    @property
    def processed_dir(self):
        return self.output_files_dir

    def process(self):
        if self.parse_metadata:
            graph_metadata = {"edge_types": set(), "node_types": set()}
            for networkx_file_path in tqdm(self.raw_paths, desc="Parsing pytorch geometric graphs:"):
                networkx_graph = networkx.read_gpickle(networkx_file_path)
                for edge in networkx_graph.edges.data():
                    node_1_type = networkx_graph.nodes[edge[0]]["type"]
                    edge_type = edge[2]['type']
                    node_2_type = networkx_graph.nodes[edge[1]]['type']

                    graph_metadata["node_types"].add(node_1_type)
                    graph_metadata["node_types"].add(node_2_type)
                    graph_metadata["edge_types"].add((node_1_type, edge_type, node_2_type))
        else:
            if self.reverse_edges:
                graph_metadata = REVERSED_GRAPH_METADATA
            else:
                graph_metadata = GRAPH_METADATA

        # Convert networkx graphs to pyg
        convert_func = functools.partial(networkx_file_to_pyg if self.data is None else graph_to_pyg, 
                                         node_featurizer = self.node_featurizer, 
                                         edge_featurizer = self.edge_featurizer, 
                                         include_labels = self.include_labels,
                                         heterogenous = self.heterogenous,
                                         graph_metadata = graph_metadata,
                                         reverse_edges = self.reverse_edges)
            
        data_list = []
        if self.data is None:
            conversion_inputs = self.raw_paths
        else:
            conversion_inputs = self.data

        if self.num_workers == 1:
            for conversion_input in tqdm(conversion_inputs):
                data_list.append(convert_func(conversion_input))
        else:
            # TODO breaks for some reason
            data_list = process_map(convert_func, conversion_inputs, 
                                    total=len(conversion_inputs),
                                    max_workers=self.num_workers)
            
        if self.add_num_nodes:
            for data in data_list:
                num_nodes = data.x.size()[0]
                num_nodes_vec = num_nodes * torch.ones([data.edge_attr.size()[0], 1]) / 250.0 # divide to make it small lol
                data.edge_attr = torch.cat([data.edge_attr, num_nodes_vec], dim=1)

        if self.add_num_edges:
            for data in data_list:
                num_edges = data.edge_attr.size()[0]
                num_edges_vec = num_edges * torch.ones([data.edge_attr.size()[0], 1]) / 500.0
                data.edge_attr = torch.cat([data.edge_attr, num_edges_vec], dim=1)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def make_featurizers(cfg, for_data_collection, steps_per_scene=100, scene_avg_num_nodes=200, scene_avg_num_edges=400):
    if cfg.model_name == 'hgtt':
        if 'node_type' in node_featurizer.features:
            node_featurizer.features.remove('node_type')
            node_featurizer.features_size-=5
        if 'edge_type' in edge_featurizer.features:
            edge_featurizer.features.remove('edge_type')
        edge_featurizer.features_size-=5
    node_featurizer = NodeFeaturizer(cfg.node_features,
                                     steps_per_scene = steps_per_scene,
                                     scene_avg_num_nodes = scene_avg_num_nodes,
                                     embed_text_with_transformer = cfg.embed_text_with_transformer)
    edge_featurizer = EdgeFeaturizer(cfg.edge_features, 
                                     node_featurizer = node_featurizer,
                                     include_labels = for_data_collection,
                                     zero_priors_to_objects = cfg.zero_priors_to_objects,
                                     steps_per_scene = steps_per_scene,
                                     scene_avg_num_edges = scene_avg_num_edges)
    return node_featurizer, edge_featurizer
