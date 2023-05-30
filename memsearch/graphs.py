import random
import pickle
import uuid
import logging
import matplotlib.pyplot as plt
import networkx as nx
from memsearch.util import sample_from_dict
from enum import Enum

class NodeType(Enum):
    HOUSE = "house"
    FLOOR = "floor"
    ROOM = "room"
    FURNITURE = "furniture"
    OBJECT = "object"

class EdgeType(Enum):
    ONTOP = "onTop"
    IN = "in"
    CONTAINS = "contains"
    UNDER = "under"
    CONNECTED = "connected"

RECIPROCAL_EDGE_TYPES = {
    EdgeType.ONTOP: EdgeType.UNDER,
    EdgeType.UNDER: EdgeType.ONTOP,
    EdgeType.IN: EdgeType.CONTAINS,
    EdgeType.CONTAINS: EdgeType.IN,
    EdgeType.CONNECTED : EdgeType.CONNECTED
}

def edge_enum_from_str(edge_str):
    for edge_type in EdgeType:
        if edge_type.value == edge_str:
            return edge_type

def save_graph(graph, path):
    with open(path,'wb') as graph_file:
        pickle.dump(graph,graph_file)

def load_graph(path):
    with open(path,'rb') as graph_file:
        return pickle.load(graph_file)

def load_priors_graph(priors_type):
    prior_graph = None
    if priors_type == 'detailed':
        prior_graph = load_graph('priors/detailed_prior_graph.pickle')
    elif priors_type == 'coarse':
        prior_graph = load_graph('priors/coarse_prior_graph.pickle')
    else:
        raise ValueError(f'{priors_type} is not a valid type of prior')
    return prior_graph

def snake_case_to_pretty(string):
    if 't_v' in string:
        string = string.replace('t_v','TV')
    if 'c_d' in string:
        string = string.replace('c_d','CD')
    string = string.capitalize()
    if '_' in string:
        string = ' '.join([x.capitalize() for x in string.split('_')])
    return string

def graph_to_csv(graph):
    # currently hacked to not produce CSV
    lines = []
    top_node = graph.get_house_node()
    for floor_node in top_node.get_children_nodes():
        for room_node in floor_node.get_children_nodes():
            for furniture_node in room_node.get_children_nodes():
                for object_node in furniture_node.get_children_nodes():
                    edge = furniture_node.get_edge_to(object_node)
                    line='%s,%s,%s,%s,%.4f'%(snake_case_to_pretty(room_node.label),
                                           snake_case_to_pretty(furniture_node.label),
                                           snake_case_to_pretty(object_node.label),
                                           'in' if edge.type == EdgeType.IN else 'on',
                                           edge.prob)

                    line='%s being %s %s within %s has %.4f probability'%(
                                                 snake_case_to_pretty(object_node.label),
                                                 'in' if edge.type == EdgeType.IN else 'on',
                                                 snake_case_to_pretty(furniture_node.label),
                                                 snake_case_to_pretty(room_node.label),
                                                 edge.prob)

                    lines.append(line)
    random.shuffle(lines)
    graph_str = 'room,furniture,object,relation,probability\n'
    for line in lines:
        graph_str+=line+'\n'
    return graph_str

def nodes_to_str(parent_node, root_nodes, indent=0):
    nodes_str=''
    for node in root_nodes:
      nodes_str+='  ' * indent + str(node)
      if parent_node is not None:
          edge = parent_node.get_edge_to(node)
          nodes_str = nodes_str+' '+edge.type.value
          if edge.prob is not None:
              nodes_str=nodes_str+' '+str(edge.prob)[:4]
      nodes_str+='\n'
      nodes_str+=nodes_to_str(node, node.get_children_nodes(), indent+1)
    return nodes_str

class Graph(object):
    """
    Little class to bundle functionality for reasoning over graphs
    """
    def __init__(self, nodes=[]):
        self.nodes = []
        for node in nodes:
            self.add_node(node)

    def add_node(self, node):
        self.nodes.append(node)

    def remove_node(self, node, remove_edges = True):
        self.nodes.remove(node)
        if remove_edges:
            for parent_node in node.get_parent_nodes():
                parent_node.remove_edges_to(node, remove_from_other_node=False)
            for child_node in node.get_children_nodes():
                child_node.remove_edges_to(node, remove_from_other_node=False)

    def get_node_num(self, node):
        return self.nodes.index(node)

    def get_node_by_num(self, node_num):
        return self.nodes[node_num]

    def get_node_with_unique_id(self, unique_id):
        for node in self.nodes:
            if node.unique_id == unique_id:
                return node
        return None

    def get_node_with_label(self, label, category=None):
        for node in self.nodes:
            if node.label == label and (category is None or node.category == category):
                return node
        return None

    def get_nodes_with_label(self, label):
        return [node for node in self.nodes if node.label == label]

    def get_nodes_with_description(self, description):
        return [node for node in self.nodes if node.description == description]

    def get_node_with_description(self, description):
        nodes = self.get_nodes_with_description(description)
        if len(nodes)==0:
            return None
        return nodes[0]

    def get_nodes_with_category(self, node_category):
        return [node for node in self.nodes if node.category == node_category]

    def get_node_with_category(self, node_category):
        nodes = self.get_nodes_with_category(node_category)
        if len(nodes)==0:
            return None
        return nodes[0]

    def get_nodes_with_type(self, node_type):
        return self.get_nodes(node_type)

    # TODO add support for category, label, refactor the existing get_with_ funcs to use this
    def get_nodes(self, node_type=None, description=None):
        if node_type!=None:
            nodes = [node for node in self.nodes if node.type == node_type]
        else:
            nodes = self.nodes
        if description is not None:
            nodes = [node for node in self.nodes if node.description == description]
        return nodes

    def get_object_node_move_probs(self):
        object_move_probs = {node:node.move_freq for node in self.get_nodes_with_type(NodeType.OBJECT)}
        total_prob = sum(object_move_probs.values())
        for key in object_move_probs:
            object_move_probs[key]/=total_prob
        return object_move_probs

    def sample_node(self, node_type=None):
        nodes = self.get_nodes(node_type)
        return random.choice(nodes)

    def sample_node_with_category(self, node_category):
        choices = [node for node in self.nodes if node.category == node_category]
        if len(choices) > 0:
            return random.choice(choices)
        else:
            raise ValueError('No node with category %s'%node_category)

    def has_node_with_category(self, node_category):
        for node in self.nodes:
            if node.category == node_category:
                return True
        return False

    def copy(self):
        return Graph([node.copy() for node in self.nodes])

    def get_house_node(self):
        return self.get_nodes(NodeType.HOUSE)[0]

    def __str__(self):
        return nodes_to_str(None, [self.get_house_node()])

    def get_edges(self):
        edges = set()
        for node in self.nodes:
            edges.update(node.edges)
        return edges

    @property
    def edges(self):
        return self.get_edges()

    def get_as_plottable_networkx(self, plot_type):
        G=nx.Graph()

        total_width = 50000.0
        room_nodes = sorted(self.get_nodes(NodeType.ROOM), key = lambda n: n.unique_id)
        furniture_nodes = sorted(self.get_nodes(NodeType.FURNITURE), key = lambda n: n.unique_id)
        object_nodes = sorted(self.get_nodes(NodeType.OBJECT), key = lambda n: n.unique_id)
        room_num = furniture_num = object_num = 0
        num_assignments = {}

        room_space_increment = int(total_width/(len(room_nodes)-1))
        furniture_space_increment = int(total_width/(len(furniture_nodes)-1))
        if len(object_nodes) == 1:
            object_space_increment = 0
        else:
            object_space_increment = int(total_width/(len(object_nodes)-1))

        ordered_nodes = []
        for room_node in room_nodes:
            ordered_nodes.append(room_node)
            num_assignments[room_node] = room_num
            room_num+=1
            furniture_nodes = [edge.node2 for edge in room_node.get_edges_from_me()]
            furniture_nodes = sorted(furniture_nodes, key = lambda n: n.unique_id)
            for furniture_node in furniture_nodes:
                if furniture_node in num_assignments:
                    continue
                ordered_nodes.append(furniture_node)
                num_assignments[furniture_node] = furniture_num
                furniture_num+=1
                object_nodes = [edge.node2 for edge in furniture_node.get_edges_from_me()]
                object_nodes = sorted(object_nodes, key = lambda n: n.unique_id)
                for object_node in object_nodes:
                    if object_node in num_assignments:
                        continue
                    ordered_nodes.append(object_node)
                    num_assignments[object_node] = object_num
                    object_num+=1
        node_sizes = []
        node_colors = []
        plotted_nodes = []
        node_to_label = {}
        label_counts = {node.label:0 for node in self.nodes}

        color_randomizer = random.Random()
        for node in ordered_nodes:
            node_num = num_assignments[node]
            if node.type == NodeType.ROOM:
                pos = (node_num * room_space_increment, 30)
                node_sizes.append(500.0)
            elif node.type == NodeType.FURNITURE:
                node_sizes.append(150.0)
                pos = (node_num * furniture_space_increment, 20)
            elif node.type == NodeType.OBJECT:
                node_sizes.append(50.0)
                height = 10
                if node_num % 2 == 1:
                    height+=2
                pos = (node_num * object_space_increment, height)
            color_randomizer.seed(node.label)
            color = '#%02X%02X%02X' % (color_randomizer.randint(0,255),
                                       color_randomizer.randint(0,255),
                                       color_randomizer.randint(0,255))
            node_colors.append(color)
            label_counts[node.label]+=1
            label = node.label + str(label_counts[node.label])
            node_to_label[node] = label
            G.add_node(label, pos=pos)
            plotted_nodes.append(node)

        for node in plotted_nodes:
            for edge in node.get_edges_from_me():
                node1_label = node_to_label[edge.node1]
                node2_label = node_to_label[edge.node2]
                if plot_type == 'priors' and edge.prob is not None:
                    G.add_edge(node1_label, node2_label, weight=edge.prob)
                elif plot_type == 'last_outputs':
                    G.add_edge(node1_label, node2_label, weight=edge.last_sgm_prob)
                elif plot_type == 'times_observed':
                    G.add_edge(node1_label, node2_label, weight=edge.times_observed)
                elif plot_type == 'freq_true':
                    G.add_edge(node1_label, node2_label, weight=edge.freq_true)
                elif plot_type == 'times_true':
                    G.add_edge(node1_label, node2_label, weight=edge.times_state_true)
                elif plot_type == 'from_priors':
                    G.add_edge(node1_label, node2_label, weight=int(edge.from_priors))
                elif plot_type == 'from_observed':
                    G.add_edge(node1_label, node2_label, weight=1-int(edge.from_priors))
                elif plot_type == 'edge_age':
                    G.add_edge(node1_label, node2_label, weight=float(edge.age))
                elif plot_type == 'node_moves':
                    G.add_edge(node1_label, node2_label, weight=float(edge.node2.times_moved))
                else:
                    G.add_edge(node1_label, node2_label, weight=0.0)
        room_nodes = [node_to_label[node] for node in plotted_nodes if node.type==NodeType.ROOM]
        furniture_nodes = [node_to_label[node] for node in plotted_nodes if node.type==NodeType.FURNITURE]
        object_nodes = [node_to_label[node] for node in plotted_nodes if node.type==NodeType.OBJECT]
        pos = nx.get_node_attributes(G,'pos') 
        print(G)
        return G, pos, node_colors, node_sizes

    def save_png(self, path, colorize_edges = True, draw_edge_probs = False, plot_type='priors'):
        G, pos, node_colors, node_sizes = self.get_as_plottable_networkx(plot_type)

        if colorize_edges:
            edges, edge_weights = zip(*nx.get_edge_attributes(G,'weight').items())
            edge_cmap = plt.cm.Greys
            #edge_cmap = plt.cm.cividis
        else:
            edges = list(G.edges())
            edge_weights = 'k'
            edge_cmap = None

        nx.draw(G,
                pos,
                node_size=node_sizes,
                width=1.5,
                alpha=0.8,
                font_size=4,
                font_color='w',
                with_labels=False,
                node_color=node_colors,
                edgelist=edges,
                edge_color=edge_weights,
                edge_cmap=edge_cmap)

        if draw_edge_probs:
            labels = dict([((u,v,), str(d['weight'])[0:4]) for u,v,d in G.edges(data=True)])
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

        fig = plt.gcf()
        fig.set_size_inches((10, 5), forward=False)
        fig.savefig(path, dpi=250)
        fig.clear()

    def save_pngs(self, path, types=['state', 'edge_age','node_moves']):
        for plot_type in types:
            plot_path = path.replace('scene_',f'scene_{plot_type}_').lower()
            colorize_edges = (plot_type != 'state')
            self.save_png(plot_path, colorize_edges=colorize_edges, plot_type=plot_type)

class Node(object):
    """
    Basic OOP Node class. Stores edges to other nodes and to itself.
    Assumes multiple edges may be stored to other nodes (of different types).
    """
    def __init__(self,
                 label,
                 node_type,
                 category=None,
                 edges=None,
                 unique_id=None,
                 max_count=1,
                 move_freq=1.0,
                 remove_prob=0.0,
                 description=None):
        self.label = label
        self.category = category
        if unique_id is None:
            self.unique_id = label+' '+str(uuid.uuid4())
        else:
            self.unique_id = unique_id
        if edges is None:
            self.edges = []
        else:
            self.edges = edges
        self.type = node_type
        self.no_longer_in_graph = False
        self.move_freq = move_freq
        self.min_count = 0
        self.max_count = max_count
        self.remove_prob = remove_prob
        self.times_moved = 0
        if description is None:
            self.description = label
        else:
            self.description = description

    def reset_edges(self):
        self.edges = []

    def get_edges(self):
        return self.edges

    def remove_edge(self, edge):
        self.edges.remove(edge)

    def get_edges_from_me(self):
        return [edge for edge in self.edges if edge.node1 == self]

    def get_edges_to_me(self):
        return [edge for edge in self.edges if edge.node2 == self]

    def add_edge(self, edge):
        self.edges.append(edge)
        other_node = edge.node2 if edge.node2!=self else edge.node1

    def get_edges_to(self, node):
        return [edge for edge in self.edges if edge.node1==node or edge.node2==node]

    def remove_edges_to(self, node, remove_from_other_node=True):
        edges_to_remove = self.get_edges_to(node)
        for edge_to_remove in edges_to_remove:
            self.edges.remove(edge_to_remove)
        if remove_from_other_node:
            node.remove_edges_to(self, False)

    def get_edge_to(self, node):
        edges = self.get_edges_to(node)
        if len(edges) == 0:
            return None
        return edges[0]

    def add_edge_from(self, node, edge_type, prob=None):
        edge = Edge(node, self, edge_type, prob)
        return edge

    def add_edge_to(self, node, edge_type):
        edge = Edge(self, node, edge_type)
        self.add_edge(edge)
        node.add_edge(edge)

    def has_edge_to(self, node, edge_type):
        return len([e for e in self.get_edges_to(node) if e.type==edge_type])!=0

    def has_edges_to(self, node):
        return len(self.get_edges_to(node))!=0

    def get_neighbor_nodes(self):
        nodes = set()
        for edge in self.edges:
            nodes.add(edge.node2)
        return list(nodes)

    def get_parent_nodes(self):
        nodes = set()
        for edge in self.edges:
            if self == edge.node2:
                nodes.add(edge.node1)
        return list(nodes)

    def get_parent_node(self):
        return self.get_parent_nodes()[0]

    def get_children_nodes(self):
        nodes = set()
        for edge in self.edges:
            if self == edge.node1:
                nodes.add(edge.node2)
        return list(nodes)

    def get_parent(self):
        """
        Assuming there's a single node with an edge to this one, return that
        """
        for edge in self.edges:
            if edge.type == EdgeType.IN or edge.type == EdgeType.ONTOP:
                return edge.node1

    def normalize_edges_to_me(self):
        total_prob = 0
        for parent_edge in self.get_edges_to_me():
            total_prob+=parent_edge.prob
        if total_prob == 0:
            for parent_edge in self.get_edges_to_me():
                parent_edge.prob = 0
        else:
            for parent_edge in self.get_edges_to_me():
                parent_edge.prob = parent_edge.prob / total_prob

    def normalize_edges_from_me(self, rescale=False):
        total_prob = 0
        max_prob = 0
        min_prob = 0
        for child_edge in self.get_edges_from_me():
            total_prob+=child_edge.prob
            max_prob = max(max_prob, child_edge.prob)
            min_prob = min(min_prob, child_edge.prob)
        if total_prob == 0:
            for child_edge in self.get_edges_from_me():
                child_edge.prob = 0
        else:
            for child_edge in self.get_edges_from_me():
                if rescale:
                    child_edge.prob = (child_edge.prob - min_prob)/(max_prob - min_prob)
                else:
                    child_edge.prob = child_edge.prob / total_prob

    def __str__(self):
        return self.unique_id

    def copy(self, with_unique_id = True, with_edge_copies = False, exclude_edges = False, to_priors_node = False):
        node_class = Node
        if to_priors_node:
            node_class = PriorsNode

        edges = None if exclude_edges or with_edge_copies else self.edges
        unique_id = self.unique_id if with_unique_id else None
        copy_node = node_class(self.label, self.type, self.category, edges,
                    unique_id, self.max_count, self.move_freq, 
                    self.remove_prob, self.description)

        if with_edge_copies:
            for edge in self.edges:
                copy_edge = edge.copy()
                if edge.node1 == self:
                    copy_edge.node1 = copy_node
                else:
                    copy_edge.node2 = copy_node
                copy_edge.add_to_nodes()

        return copy_node

    def copy_no_edges(self, with_unique_id = True):
        return self.copy(with_unique_id = with_unique_id, exclude_edges=True)

    def copy_to_sgm(self, from_observation=True):
        # No edges since SGM edges are different from scene graph edges
        return SGMNode(self.label, self.type, self.category, None, self.unique_id, self.description, from_observation)

    def copy_to_priors(self, with_unique_id = True):
        return self.copy(with_unique_id=with_unique_id, exclude_edges=True, to_priors_node=True)

class Edge(object):

    def __init__(self,
                 node1,
                 node2,
                 edge_type,
                 prob=None,
                 directed=True,
                 add_to_nodes=True):
        self.node1 = node1
        self.node2 = node2
        self.type = edge_type
        self.prob = prob
        self.directed = directed
        self.time_since_observed = -1
        self.times_observed = 0
        self.added = False
        self.age = 0
        self.last_sgm_prob = prob
        if add_to_nodes:
            self.add_to_nodes()

    def add_to_nodes(self):
        if not self.added:
            self.added = True
            self.node1.add_edge(self)
            self.node2.add_edge(self)
        else:
            logging.warning('Tried to add edge to nodes a second time')
            raise RuntimeError('Cant add edge twice')

    def remove_from_nodes(self):
        self.node1.remove_edge(self)
        self.node2.remove_edge(self)

    def copy(self, add_to_nodes=False):
        return Edge(self.node1, self.node2, self.type, self.prob, self.directed, add_to_nodes=add_to_nodes)

    def __str__(self):
        if self.prob is not None:
            return '%s from %s to %s with prob %.3f'%(self.type, str(self.node1), str(self.node2), self.prob)
        else:
            return '%s from %s to %s'%(self.type, str(self.node1), str(self.node2))


class PriorsGraph(Graph):
    def __init__(self, nodes):
        super().__init__(nodes)

    def get_edges_to(self, node_label, node_type=None):
        nodes = self.get_nodes_with_label(node_label)
        edges = []
        for node in nodes:
            for node_edge in node.edges:
                if node_type is None or node_edge.node1.type == node_type:
                    edges.append(node_edge)
        return edges

    def sample_edge_to(self, node):
        edge_options = self.get_edges_to(node, NodeType.FURNITURE)
        edge_dict = {edge:edge.prob for edge in edge_options}
        choice = random.choices(list(edge_dict.keys()), weights=edge_dict.values(), k=1)[0]
        return choice

class PriorsNode(Node):
    def __init__(self,
                 label,
                 node_type,
                 category=None,
                 edges=None,
                 unique_id=None,
                 max_count=1,
                 move_freq=1.0,
                 description=None,
                 adjectives=None,
                 sample_prob=1.0,
                 spawn_prob=0.0):
        super().__init__(label,
                         node_type,
                         category,
                         edges,
                         unique_id,
                         max_count,
                         move_freq,
                         description)
        self.adjectives = adjectives
        self.sample_prob = sample_prob
        self.spawn_prob = spawn_prob
        self.remove_prob = spawn_prob

    def sample_edge(self):
        options_dict = {}
        for edge in self.edges:
            if self==edge.node2 and edge.prob!=None:
                options_dict[edge] = edge.prob
        if len(options_dict)==0:
            return None
        choice = sample_from_dict(options_dict)
        return choice

    def sample_children_nodes(self, min_sampled=1, max_sampled=1):
        nodes = []
        nodes_with_probs = {node:node.sample_prob for node in self.get_children_nodes()}
        node_counts = {node:0 for node in self.get_children_nodes()}
        give_up_count = 0
        num_to_sample = random.randint(min_sampled, max_sampled)
        while len(nodes) != num_to_sample:
            give_up_count+=1

            node = sample_from_dict(nodes_with_probs)
            if node_counts[node] > node.max_count and give_up_count < 100:
                # ignore max count if we have too... rare edge case
                continue

            node_counts[node]+=1
            nodes.append(node)
        return nodes

    def copy(self, with_unique_id = True, with_edge_copies = False, exclude_edges=False):
        copy_node = super().copy(with_unique_id, with_edge_copies, exclude_edges, to_priors_node=True)
        copy_node.adjectives = self.adjectives
        copy_node.sample_prob = self.sample_prob
        copy_node.spawn_prob = self.spawn_prob
        copy_node.remove_prob = self.spawn_prob
        return copy_node

    def copy_to_normal_node(self, with_unique_id = False, with_edge_copies = False, exclude_edges=True):
        return super().copy(with_unique_id, with_edge_copies, exclude_edges, to_priors_node=False)

class SGM(Graph):
    """
    Little subclass to bundle functionality specific to SGM stuff
    (very little for now, might move stuff from Agent into here)
    """
    def __init__(self, nodes=[]):
        super().__init__(nodes)

    def step(self):
        for node in self.nodes:
            node.time_since_observed+=1
            node.time_since_state_change+=1
        for edge in self.get_edges():
            edge.time_since_observed+=1
            edge.time_since_state_change+=1

    def save_pngs(self, path, types=['priors','last_outputs','times_observed','freq_true','times_true','from_priors','from_observed']):
        for plot_type in types:
            sgm_model_type = list(self.edges)[0].sgm_model
            plot_path = path.replace('sgm_',f'sgm_{plot_type}_{sgm_model_type}_').lower()
            self.save_png(plot_path, plot_type=plot_type)

    def copy(self):
        graph = SGM([node.copy() for node in self.nodes])
        for edge in self.get_edges():
            new_node1 = graph.get_node_with_unique_id(edge.node1.unique_id)
            new_node2 = graph.get_node_with_unique_id(edge.node2.unique_id)
            edge_copy = edge.copy()
            edge_copy.node1 = new_node1
            edge_copy.node2 = new_node2
            edge_copy.add_to_nodes()
        return graph

class SGMNode(Node):
    """
    A class to represent Nodes in a SGM
    """
    def __init__(self,
                 label,
                 node_type,
                 category=None,
                 edges=None, #edges are stored as dicts with keys of ids to nodes
                 unique_id=None,
                 description=None,
                 from_observation=False):
        super().__init__(label, node_type, category, edges, unique_id, description=description)
        self.label = label
        self.time_since_observed = -1
        self.times_observed = 0
        self.last_sgm_prob = 0
        self.time_since_state_change = -1
        self.times_state_changed = 0
        self.from_observation = from_observation

    @property
    def state_change_freq(self):
        if self.times_observed == 0:
            return 0
        return float(self.times_state_changed)/self.times_observed

    def observe(self, parent_description=None, state_has_changed=False):
        self.time_since_observed = 0
        self.times_observed+= 1
        if parent_description is not None:
            self.parent_description = parent_description
        if state_has_changed:
            self.time_since_state_change = 0
            self.times_state_changed+=1

    def normalize_sgm_probs(self):
        total_prob = 0
        for parent_edge in self.get_edges_to_me():
            total_prob+=parent_edge.last_sgm_prob
        for parent_edge in self.get_edges_to_me():
            parent_edge.last_sgm_prob = parent_edge.last_sgm_prob / total_prob

    def copy(self, with_unique_id = True, with_edge_copies = False, exclude_edges = True):
        edges = None if exclude_edges or with_edge_copies else self.edges
        unique_id = self.unique_id if with_unique_id else None
        copy_node = SGMNode(self.label, self.type, self.category, edges,
                    unique_id, self.description, self.from_observation)

        if with_edge_copies:
            for edge in self.edges:
                copy_edge = edge.copy()
                if edge.node1 == self:
                    copy_edge.node1 = copy_node
                else:
                    copy_edge.node2 = copy_node
                copy_edge.add_to_nodes()

        copy_node.time_since_observed = self.time_since_observed
        copy_node.times_observed = self.times_observed
        copy_node.last_sgm_prob = self.last_sgm_prob
        copy_node.time_since_state_change = self.time_since_state_change
        copy_node.times_state_changed = self.times_state_changed

        return copy_node

    def __str__(self):
        return str(self.category)+' '+self.unique_id

class SGMEdge(Edge):

    def __init__(self,
                 node1,
                 node2,
                 edge_type,
                 prob,
                 add_to_nodes=True,
                 from_priors=False):
        super().__init__(node1, node2, edge_type, prob, directed=True, add_to_nodes=add_to_nodes)
        self.from_priors = from_priors
        self.time_since_observed = -1
        self.times_observed = 0
        self.times_state_true = 0
        self.times_state_changed = 0
        self.time_since_observed_state = False
        self.time_since_state_change = -1
        self.currently_true = False
        self.last_observed_state = False
        self.is_query_edge = False
        self.sgm_model = 'gcn' #will be overwritten in agent

    @property
    def freq_true(self):
        if self.times_observed == 0:
            return 0.1
        return float(self.times_state_true)/self.times_observed

    @property
    def state_change_freq(self):
        if self.times_observed == 0:
            return 0.1
        return float(self.times_state_changed)/self.times_observed

    def observe(self, state, set_prob_to_one=True):
        self.time_since_observed = 0
        self.times_observed+=1

        if state and set_prob_to_one:
            self.prob = 1.0

        if state:
            self.times_state_true+=1

        if state!=self.last_observed_state:
            self.time_since_state_changed = 0

        self.last_observed_state = state

    def __str__(self):
        return '%s from %s to %s with prob %.2f'%(self.type,
                str(self.node1), str(self.node2), self.prob)

    def copy(self, add_to_nodes=False):
        copy_edge = SGMEdge(self.node1, self.node2, self.type, self.prob, add_to_nodes=add_to_nodes, from_priors=self.from_priors)
        copy_edge.time_since_observed = self.time_since_observed
        copy_edge.times_observed = self.times_observed
        copy_edge.times_state_true = self.times_state_true
        copy_edge.times_state_changed = self.times_state_changed
        copy_edge.time_since_observed_state = self.time_since_observed_state
        copy_edge.time_since_state_change = self.time_since_state_change
        copy_edge.currently_true = self.currently_true
        copy_edge.last_observed_state = self.last_observed_state
        copy_edge.is_query_edge = self.is_query_edge
        copy_edge.sgm_model = self.sgm_model#will be overwritten in agent
        return copy_edge

if __name__ == "__main__":
    prior_graph = load_graph('data/prior_graph.pickle')
    scene_graph = prior_graph.sample_scene_graph()
    print(nodes_to_str([scene_graph.get_house_node()]))
    scene_graph.save_png('scene_graph.png')
