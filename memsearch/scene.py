"""
Everything related to scene representation, sampling, and evolution.
"""
from collections import defaultdict
from memsearch.graphs import *
from memsearch.util import sample_from_dict
import numpy as np
import logging
import random

DEFAULT_ROOM_LAYOUT = {
    'bedroom': {'bed': 1,
                'desk': 1,
                'chair': 1
                },
    'kitchen': {'fridge': 1,
                'counter_top': 2,
                # 'can': 1
                },
    'bathroom': {'toilet': 1,
                 'sink': 1
                 },
    'living_room': {'chair': 10,
                    'dining_table': 1,
                    'shelving_unit': 1,
                    'sofa': 1
                    }
}

class Scene(object):
    """
    A class to deal with logic of a scene.
    Wraps a Scene Graph and has additional util logic.
    """
    def __init__(self, scene_graph):
        self.scene_graph = scene_graph
        self.time = 0

    def move_object(self, obj_node, from_node, to_node, relation_type):
        from_node.remove_edges_to(obj_node, remove_from_other_node=True)
        
        # Original Edge
        Edge(to_node, obj_node, relation_type)

    def get_graph(self):
        return self.scene_graph

    def sample_object(self):
        return self.scene_graph.sample_node(NodeType.OBJECT)

    def sample_furniture(self):
        return self.scene_graph.sample_node(NodeType.FURNITURE)

class SceneSampler(object):
    """
    A class that deals with sampling scenes
    """
    def __init__(self):
        pass

    def sample(self):
        raise NotImplementedError()

def create_node_description(node, possible_adjectives):
    if possible_adjectives is None or len(possible_adjectives) == 0: #TODO figure out why this is ever the case.
        return node.label
    if type(possible_adjectives[0]) is str:
        possible_adjectives = [possible_adjectives]
    description = node.label
    add_num_adjectives = random.randint(1,len(possible_adjectives))
    num_adjectives_added = 0
    for adjectives_list in reversed(possible_adjectives):
        description = random.choice(adjectives_list) + '_' + description
        num_adjectives_added+=1
        if num_adjectives_added == add_num_adjectives:
            break
    return description

class PriorsSceneSampler(object):
    """
    A class that deals with sampling scenes
    """
    def __init__(self,
                 base_priors_graph,
                 min_floors=1,
                 max_floors=1,
                 min_rooms=4,
                 max_rooms=6,
                 min_furniture=3,
                 max_furniture=6,
                 min_objects=4,
                 max_objects=12,
                 priors_noise=0,
                 sparsity_level=0.1,
                 room_layout=None):
        self.base_priors_graph = base_priors_graph
        self.current_priors_graph = base_priors_graph
        self.priors_noise = priors_noise
        self.min_floors = min_floors
        self.max_floors = max_floors
        self.min_rooms = min_rooms
        self.max_rooms = max_rooms
        self.min_furniture = min_furniture
        self.max_furniture = max_furniture
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.sparsity_level = sparsity_level
        self.node_label_counts = defaultdict(lambda: 0)
        self.scene_num = 0
        self.room_layout = room_layout

    def sample(self):
        if self.priors_noise != 0 or self.sparsity_level != 0:
            self.current_priors_graph = copy_priors_with_noise(priors_graph = self.base_priors_graph, 
                                                       scene_random_seed=random.random(),
                                                       normalize_probs_to_nodes=False,
                                                       normalize_probs_from_nodes=False,
                                                       sparsity_level=self.sparsity_level,
                                                       priors_noise=self.priors_noise)
        else:
            self.current_priors_graph = self.base_priors_graph
        return Scene(self.sample_scene_graph())

    def sample_scene_graph(self):
        self.scene_num+=1
        priors_graph = self.current_priors_graph
        self.node_label_counts = defaultdict(lambda: 0)

        house_node = priors_graph.get_nodes(NodeType.HOUSE)[0]
        priors_floor_node = house_node.get_children_nodes()[0]
        #priors_floor_nodes = house_node.sample_children_nodes(min_sampled=self.min_floors,
        #                                               max_sampled=self.max_floors)

        new_house_node = house_node.copy_to_normal_node()
        new_house_node.description = 'house'
        scene_graph_nodes = [new_house_node]
        
        #lables and categories same for now
        scene_graph_floor_node = priors_floor_node.copy_to_normal_node()
        scene_graph_floor_node.add_edge_from(new_house_node, EdgeType.CONTAINS)
        scene_graph_nodes.append(scene_graph_floor_node)

        if self.max_rooms >= 4:
            priors_room_nodes = priors_floor_node.get_children_nodes()
        else:
            priors_room_nodes = self.sample_children_nodes(priors_floor_node, self.min_rooms, self.max_rooms)
        for priors_room_node in priors_room_nodes:
            scene_graph_room_node = priors_room_node.copy_to_normal_node()
            scene_graph_room_node.description = scene_graph_room_node.label
            scene_graph_room_node.add_edge_from(scene_graph_floor_node, EdgeType.CONTAINS)
            scene_graph_nodes.append(scene_graph_room_node)

            if self.room_layout is None:
                priors_furniture_nodes = self.sample_children_nodes(priors_room_node, self.min_furniture, self.max_furniture)
            else:
                room_layout = self.room_layout[scene_graph_room_node.label]
                room_children_nodes = priors_room_node.get_children_nodes()
                priors_furniture_nodes = []
                for obj_type, num in room_layout.items():
                    obj_node = [node for node in room_children_nodes if node.label == obj_type][0]
                    priors_furniture_nodes += [obj_node for _ in range(num)]

            for priors_furniture_node in priors_furniture_nodes:
                if priors_furniture_node.type != NodeType.FURNITURE:
                    #just in case
                    continue
                scene_graph_furniture_node = priors_furniture_node.copy_to_normal_node()
                scene_graph_furniture_node.add_edge_from(scene_graph_room_node, EdgeType.CONTAINS)
                scene_graph_nodes.append(scene_graph_furniture_node)
                scene_graph_furniture_node.description = create_node_description(scene_graph_furniture_node, priors_furniture_node.adjectives)
                
                if self.room_layout and 'chair' in priors_furniture_node.category:
                    continue 
                
                priors_object_nodes = self.sample_children_nodes(priors_furniture_node, self.min_objects, self.max_objects)
                assert len(priors_object_nodes) > 0, 'Cant sample object for {furniture_node.category}'

                for priors_object_node in priors_object_nodes:
                    possible_edges = priors_object_node.get_edges_to(priors_furniture_node)
                    if len(possible_edges) == 0:
                        logging.error("Child object node has no edges to parent furniture node. Skipping")
                        continue

                    edge_type = random.choice(possible_edges).type
                    new_object_node = priors_object_node.copy_to_normal_node()
                    new_object_node.description = create_node_description(new_object_node, priors_object_node.adjectives)

                    # Original edge
                    new_object_node.add_edge_from(scene_graph_furniture_node, edge_type)

                    scene_graph_nodes.append(new_object_node)

        return Graph(scene_graph_nodes)

    def sample_children_nodes(self, node, min_sampled=1, max_sampled=1):
        nodes_with_probs = {}
        valid_edges = node.get_edges_from_me() if self.room_layout else node.edges
        for edge in valid_edges:
            nodes_with_probs[edge.node2] = edge.prob #*edge.node2.sample_prob
        give_up_count = 0
        nodes = []
        num_to_sample = random.randint(min_sampled, max_sampled)
        while len(nodes) != num_to_sample:
            give_up_count += 1
            node = sample_from_dict(nodes_with_probs)
            if self.node_label_counts[node.label] > node.max_count and give_up_count < 100:
                # ignore max count if we have too... rare edge case
                continue
            self.node_label_counts[node.label]+=1
            nodes.append(node)
        return nodes

class SceneEvolver(object):
    """
    A class that deals with simulating changes in scenes over time
    """
    def __init__(self, object_move_ratio_per_step = 1):
        self.object_move_ratio_per_step = object_move_ratio_per_step

    def set_new_scene(self, scene):
        """
        Optional method to init some logic for a new scene
        """
        self.scene = scene
        self.initial_num_objects = len(scene.scene_graph.nodes)
        num_object_nodes = len(self.scene.scene_graph.get_nodes_with_type(NodeType.OBJECT))
        self.object_moves_per_step = int(num_object_nodes*self.object_move_ratio_per_step)
        self.add_remove_prob_multiplier = 1.0/self.initial_num_objects*30.0 # hacky way to avoid a ton of addition/removal

    def get_move_target_probs(self, target_to_move):
        raise NotImplementedError()

    def evolve(self):
        raise NotImplementedError()

class RandomSceneEvolver(SceneEvolver):
    """
    Scene evolver that just moves objects around at random
    """
    def __init__(self, object_move_ratio_per_step = 1):
        super().__init__(object_move_ratio_per_step)

    def get_move_target_probs(self):
        pass

    def evolve(self):
        objects_moved = []
        for i in range(self.object_moves_per_step):
            scene_object = self.scene.sample_object()
            old_furniture = scene_object.get_furniture()
            new_furniture = self.scene.sample_furniture()
            if random.random() < 0.5:
                relation = EdgeType.CONTAINS
            else:
                relation = EdgeType.UNDER
            self.scene.move_object(scene_object, old_furniture, new_furniture, relation)
            objects_moved.append(scene_object)
        return objects_moved

class PriorsSceneEvolver(SceneEvolver):
    """
    Scene evolver that just moves objects around at random
    """
    def __init__(self,
                 base_priors_graph,
                 scene_sampler,
                 object_move_ratio_per_step=0.1,
                 use_move_freq=False,
                 prior_nodes_from_scene=False,
                 dont_double_move_object=True,
                 add_or_remove_objs=False,
                 priors_noise=0.2,
                 sparsity_level=0.1):
        super().__init__()
        self.base_priors_graph = base_priors_graph
        self.current_priors_graph = base_priors_graph
        self.object_move_ratio_per_step = object_move_ratio_per_step
        self.use_move_freq = use_move_freq
        self.prior_nodes_from_scene = prior_nodes_from_scene
        self.dont_double_move_object = dont_double_move_object
        self.scene_sampler = scene_sampler
        self.moved_objects = []
        self.add_or_remove_objs = add_or_remove_objs
        self.priors_noise = priors_noise
        self.sparsity_level = sparsity_level
        self.scene_max_objs_per_furniture = scene_sampler.max_objects

    def set_new_scene(self, scene):
        super().set_new_scene(scene)
        self.scene = scene
        if self.scene_sampler is None:
            priors_graph = self.base_priors_graph
        else:
            priors_graph = self.scene_sampler.current_priors_graph
        self.scene_random_seed = random.random()
        self.current_priors_graph = copy_priors_with_noise(priors_graph = priors_graph, 
                                                  scene_graph = scene.scene_graph, 
                                                  normalize_probs_to_nodes = True, 
                                                  scene_random_seed = self.scene_random_seed, 
                                                  sparsity_level = self.sparsity_level,
                                                  priors_noise = self.priors_noise)
        for node in self.current_priors_graph.get_nodes_with_type(NodeType.ROOM):
            for edge in node.edges:
                edge.prob = 0.1

    def get_move_target_probs(self, object_to_move, exclude_nodes=[]):
        priors_graph = self.current_priors_graph
        scene_graph = self.scene.scene_graph
        if self.prior_nodes_from_scene:
            priors_node = priors_graph.get_node_with_unique_id(object_to_move.unique_id)
        else:
            priors_node = priors_graph.get_node_with_category(object_to_move.category)

        possible_options = {}
        if priors_node is None or len(priors_node.edges)==1:
            return possible_options
        for edge in priors_node.edges:
            priors_furniture_node = edge.node1
            if self.prior_nodes_from_scene:
                furniture_nodes = [scene_graph.get_node_with_unique_id(priors_furniture_node.unique_id)]
            else:
                furniture_nodes = scene_graph.get_nodes_with_category(priors_furniture_node.category)
            for furniture_node in furniture_nodes:
                num_obj_children = len([node for node in furniture_node.get_children_nodes() if node.type == NodeType.OBJECT])
                if furniture_node not in exclude_nodes and num_obj_children + 1 <= self.scene_max_objs_per_furniture:
                    possible_options[(furniture_node, edge)] = edge.prob
        return possible_options

    def _remove_objects(self):
        objects_to_remove = []
        for object_node in self.scene.scene_graph.get_nodes_with_type(NodeType.OBJECT):
            if random.random() > object_node.remove_prob*self.add_remove_prob_multiplier*2.5:
                continue # not removing this
            objects_to_remove.append(object_node)
        for object_node in objects_to_remove:
            self.scene.scene_graph.remove_node(object_node)
            if self.prior_nodes_from_scene:
                priors_node = self.current_priors_graph.get_node_with_unique_id(object_node.unique_id)
                self.current_priors_graph.remove_node(priors_node)
        return objects_to_remove

    def _move_objects(self):
        if self.use_move_freq:
            object_move_options = self.scene.scene_graph.get_object_node_move_probs()
        else:
            object_move_options = {node:1.0 for node in self.scene.scene_graph.get_nodes_with_type(NodeType.OBJECT)}

        if self.prior_nodes_from_scene:
            remove_objects = []
            for obj in object_move_options:
                priors_obj = self.current_priors_graph.get_node_with_unique_id(obj.unique_id)
                if len(priors_obj.edges) == 1:
                    remove_objects.append(obj)

        for obj in remove_objects:
            del object_move_options[obj]

        num_moved = 0
        tries = 0
        objects_moved = []
        while num_moved < self.object_moves_per_step and tries < 100:
            object_to_move = sample_from_dict(object_move_options)
            tries +=1
            if tries == 100:
                logging.warn('Could not find a valid object to move')

            old_furniture = object_to_move.get_parent_node()
            option_probs = self.get_move_target_probs(object_to_move, [old_furniture])

            if len(option_probs) == 0:
                continue

            if not self.dont_double_move_object:
                del object_move_options[object_to_move]

            new_furniture, edge_choice = sample_from_dict(option_probs)
            num_moved+=1
            self.scene.move_object(object_to_move, old_furniture,
                              new_furniture, edge_choice.type)
            self.moved_objects.append(object_to_move)
            objects_moved.append(object_to_move)
        return objects_moved

    def _add_objects(self):
        objects_added = []
        priors_objects_nodes_added = []
        for priors_furniture_node in self.current_priors_graph.get_nodes_with_type(NodeType.FURNITURE):
            sampler_priors_furniture_node = self.scene_sampler.current_priors_graph.get_node_with_category(priors_furniture_node.category)
            scene_graph_furniture_node = self.scene.scene_graph.get_node_with_unique_id(priors_furniture_node.unique_id)
            for sampler_priors_object_node in sampler_priors_furniture_node.get_children_nodes():
                if random.random() > (sampler_priors_object_node.spawn_prob*self.add_remove_prob_multiplier): 
                    continue # not adding this
                possible_edges = sampler_priors_object_node.get_edges_to(sampler_priors_furniture_node)
                edge_type = random.choice(possible_edges).type
                new_object_node = sampler_priors_object_node.copy_to_normal_node()
                new_object_node.description = create_node_description(new_object_node, sampler_priors_object_node.adjectives)
                new_object_node.add_edge_from(scene_graph_furniture_node, edge_type)

                objects_added.append(new_object_node)
                self.scene.scene_graph.add_node(new_object_node)

                if self.prior_nodes_from_scene:
                    priors_object_node = new_object_node.copy_to_priors(with_unique_id=True)
                    priors_objects_nodes_added.append(priors_object_node)
                    self.current_priors_graph.add_node(priors_object_node)

        prior_randomizer = random.Random()
        for new_priors_object_node in priors_objects_nodes_added:
            num_zeroes = 0
            sampler_priors_node = self.scene_sampler.current_priors_graph.get_node_with_category(new_priors_object_node.category)
            priors_node_edges = list(sampler_priors_node.get_edges_to_me())
            num_possible_edges = sum([1 if edge.prob > 0.05 else 0 for edge in priors_node_edges])

            for priors_edge in priors_node_edges:
                category1 = priors_edge.node1.category
                category2 = priors_edge.node2.category

                # consistent within scene, different for different scenes
                prior_randomizer.seed(str(self.scene_random_seed)+category1+category2)

                edge_nodes1 = self.current_priors_graph.get_nodes_with_category(category1)
                for edge_node1 in edge_nodes1:
                    prob = priors_edge.prob
                    if prior_randomizer.random() < self.sparsity_level and (num_zeroes+2) <= num_possible_edges:
                        prob = 0.01
                        num_zeroes+=1
                    elif self.priors_noise > 0.0:
                        prob*= 1.0 + (prior_randomizer.random()-0.5)*2*self.priors_noise
                    Edge(edge_node1, new_priors_object_node, priors_edge.type, prob)
        for new_priors_object_node in priors_objects_nodes_added:
            new_priors_object_node.normalize_edges_to_me()
        return objects_added

    def evolve(self):
        num_nodes = len(self.scene.scene_graph.nodes)

        if self.add_or_remove_objs and num_nodes > self.initial_num_objects * 0.95:
            self._remove_objects()
            num_object_nodes = len(self.scene.scene_graph.get_nodes_with_type(NodeType.OBJECT))
            self.object_moves_per_step = int(num_object_nodes*self.object_move_ratio_per_step)

        moved_objects = self._move_objects()

        if self.add_or_remove_objs and num_nodes < self.initial_num_objects * 1.05:
            self._add_objects()

        return moved_objects

def make_scene_sampler_and_evolver(cfg):
    priors_graph = load_priors_graph(cfg.scene_priors_type)
    scene_sampler = PriorsSceneSampler(priors_graph,
                                        min_floors=cfg.min_floors,
                                        max_floors=cfg.max_floors,
                                        min_rooms=cfg.min_rooms,
                                        max_rooms=cfg.max_rooms,
                                        min_furniture=cfg.min_furniture,
                                        max_furniture=cfg.max_furniture,
                                        min_objects=cfg.min_objects,
                                        max_objects=cfg.max_objects,
                                        priors_noise=cfg.priors_noise,
                                        sparsity_level=cfg.priors_sparsity_level
                                    )
    if cfg.random_evolver:
        scene_evolver = RandomSceneEvolver()
    else:
        scene_evolver = PriorsSceneEvolver(priors_graph, 
                                           scene_sampler,
                                           object_move_ratio_per_step=cfg.object_move_ratio_per_step,
                                           prior_nodes_from_scene=cfg.evolver_priors_nodes_from_scene,
                                           use_move_freq=cfg.use_move_freq,
                                           add_or_remove_objs=cfg.add_or_remove_objs,
                                           sparsity_level=cfg.priors_sparsity_level)

    return scene_sampler, scene_evolver

def copy_priors_with_noise(priors_graph, 
                  scene_graph = None, 
                  normalize_probs_from_nodes=False, 
                  normalize_probs_to_nodes=False, 
                  print_num_possible_edges=False, 
                  scene_random_seed=0,
                  sparsity_level=0.1,
                  priors_noise=0.1):
    if scene_graph is None:
        priors_graph_copy = PriorsGraph([node.copy(with_unique_id=True, exclude_edges=True) for node in priors_graph.nodes])
    else:
        priors_graph_copy = PriorsGraph([node.copy_to_priors(with_unique_id=True) for node in scene_graph.nodes])

    prior_randomizer = random.Random()
    num_possible_edges_pre = []
    for copy_priors_node in priors_graph_copy.nodes:
        num_zeroes = 0
        priors_node = priors_graph.get_node_with_category(copy_priors_node.category)
        priors_node_edges = list(priors_node.get_edges_to_me())
        num_possible_edges = sum([1 if edge.prob > 0.0001 else 0 for edge in priors_node_edges])

        if print_num_possible_edges and sparse_priors_node.type == NodeType.OBJECT:
            num_possible_edges_pre+=[num_possible_edges]

        #random.shuffle(priors_node_edges) # just to avoid any potential patterns
        for priors_edge in priors_node_edges:
            category1 = priors_edge.node1.category
            copy_edge_nodes1 = priors_graph_copy.get_nodes_with_category(category1)
            if len(copy_edge_nodes1) == 0:
                continue
            category2 = priors_edge.node2.category

            # consistent within scene, different for different scenes
            if scene_graph is None:
                prior_randomizer.seed(str(scene_random_seed)+category1+category2)
            else:
                prior_randomizer.seed(str(scene_random_seed)+category1+category2)
                #description1 = priors_edge.node1.description
                #description2 = priors_edge.node2.description
                #prior_randomizer.seed(str(scene_random_seed)+description1+description2)
            for copy_edge_node1 in copy_edge_nodes1:
                if scene_graph is not None:
                    sg_node1 = scene_graph.get_node_with_unique_id(copy_edge_node1.unique_id)
                    sg_node2 = scene_graph.get_node_with_unique_id(copy_priors_node.unique_id)
                    edge_in_scene_graph = sg_node1.has_edges_to(sg_node2)
                else:
                    edge_in_scene_graph = False
                prob = priors_edge.prob
                if copy_edge_node1.type == NodeType.FURNITURE:
                    if not edge_in_scene_graph and \
                            prior_randomizer.random() < sparsity_level and \
                            (num_zeroes+2) <= num_possible_edges:
                        prob = 0.0
                        num_zeroes+=1
                    elif priors_noise > 0.0:
                        prob*= 1.0 + (prior_randomizer.random()-0.5)*2*priors_noise
                Edge(copy_edge_node1, copy_priors_node, priors_edge.type, prob)

        if normalize_probs_from_nodes:
            for node in priors_graph_copy.nodes:
                node.normalize_edges_from_me()
        elif normalize_probs_to_nodes:
            for node in priors_graph_copy.nodes:
                node.normalize_edges_to_me()

    if print_num_possible_edges:
        num_possible_edges_post = []
        for sparse_priors_node in priors_graph_copy.get_nodes_with_type(NodeType.OBJECT):
            num_possible_edges = sum([1 if edge.prob > 0.0001 else 0 for edge in sparse_priors_node.get_edges_to_me()])
            num_possible_edges_post+=[num_possible_edges]
        logging.info('Mean number of possible edges pre sparsity %.1f'%np.mean(num_possible_edges_pre))
        logging.info('Mean number of possible edges post sparsity %.1f'%np.mean(num_possible_edges_post))
    return priors_graph_copy

def get_edge_to_parent_node_by_type(node, type_requested, step_cost=0):
    cost = 0
    node_to_inspect = node
    while True: # sometimes a furniture node's parent is another furniture node
        cost += step_cost
        edge_to_parent = [edge for edge in node_to_inspect.edges if edge.node1.type == type_requested]
        if len(edge_to_parent) > 0:
            assert len(edge_to_parent) == 1, "Node {} belongs to {} parents of type {}".format(node, len(edge_to_parent), type_requested)
            break
        node_to_inspect = node_to_inspect.edges[0].node1
    return edge_to_parent[0], cost

def get_cost_between_nodes(starting_node, dest_node, scene_graph, edges=None):
    if edges is not None:
        raise NotImplementedError("Assigning cost by edge type is not supported yet.")
    travel_to_room_cost = 5
    room_to_furniture_cost = 1
    furniture_to_room_cost = 1
    examine_furniture_cost = 2
    total_cost = 0
    # Starting node is assumed to be a furniture node
    # dest_node is the predicted node, hence also a furniture node

    # Go from current furniture to room node
    curr_graph_node = scene_graph.get_node_with_unique_id(starting_node.unique_id)
    edge_to_room, _ = get_edge_to_parent_node_by_type(curr_graph_node, NodeType.ROOM)
    init_room_node = edge_to_room.node1
    
    # Move from goal furniture node to goal room node
    edge_to_target_room, total_room_to_furniture_cost = get_edge_to_parent_node_by_type(dest_node, NodeType.ROOM, room_to_furniture_cost)
    goal_room_node = edge_to_target_room.node1

    # Compute cost
    total_cost += furniture_to_room_cost # move agent to the room level
    if goal_room_node.description != init_room_node.description:
        total_cost += travel_to_room_cost # move to the goal room if needed
    total_cost += (total_room_to_furniture_cost + examine_furniture_cost) # move from goal room level to goal furniture level, add exam cost
    return total_cost

if __name__ == '__main__':
    prior_graph = load_graph('data/prior_graph.pickle')
    scene_graph = prior_graph.sample_scene_graph()
    scene = Scene(scene_graph)
    evolver = RandomSceneEvolver()
    print(scene_graph)
    for i in range(10):
        print('----------------------------')
        objects_moved = evolver.evolve(scene)
        for obj in objects_moved:
            print(obj.unique_id)
        print('---')
        print(scene_graph)
