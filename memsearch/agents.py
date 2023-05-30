import numpy as np
import random
import torch
from abc import abstractmethod
from scipy.stats import beta
from memsearch.graphs import *
from memsearch.models import make_model, compute_output
from memsearch.dataset import graph_to_pyg, GRAPH_METADATA, REVERSED_GRAPH_METADATA
from torch_geometric.transforms.to_undirected import ToUndirected
from memsearch.util import sample_from_dict
from memsearch.tasks import TaskType

class Agent(object):
    """
    Base class for agents, with are policies along with any
    stored state that these policies utilize.
    """

    def __init__(self, task_type):
        self.task_type = task_type
  
    def receive_observation(self, observation: Node):
        if self.task_type in [TaskType.PREDICT_LOC, TaskType.FIND_OBJECT]:
            return self.receive_predict_location_obs(observation)
        elif self.task_type == TaskType.PREDICT_ENV_DYNAMICS:
            return self.receive_predict_scene_obs(observation)
        else: 
            raise Exception(f"Unsupported task_type: {self.task}")

    def receive_predict_location_obs(self, observation):
        pass

    def receive_predict_scene_obs(self, observation):
        for node in observation:
            self.receive_predict_location_obs(node)

    def transition_to_new_scene(self, scene):
        pass

    def step(self):
        # Account for passing time (scene evolution)
        pass

    def make_predictions(self, task_type, scene, query, top_k=3, ignore_nodes_to_pred=[]):
        node_options = scene.scene_graph.get_nodes(NodeType.FURNITURE)
        
        if task_type == TaskType.PREDICT_LOC:
            if type(query) is Node:
                action = self.predict_location(query, node_options)
            else:
                action = self.predict_locations(query, node_options)
        elif task_type == TaskType.PREDICT_ENV_DYNAMICS:
            action = self.predict_env_dynamics(query, node_options, top_k=top_k)
        elif task_type == TaskType.FIND_OBJECT:
            filtered_options = [node for node in node_options if not node.unique_id in ignore_nodes_to_pred]
            if len(filtered_options) == 0:
                action = None
            else:
                action = self.predict_location(query, filtered_options, ignore_nodes=ignore_nodes_to_pred)
        else:
            raise RuntimeError(f"task_type: {task_type.value} is not supported")
        return action
    
    @abstractmethod
    def predict_location(self, target_node, node_options, ignore_nodes=[]):
        pass

    def predict_locations(self, target_nodes, node_options):
        action = {goal_node_i.description: self.predict_location(goal_node_i, node_options) for goal_node_i in target_nodes}
        return action

    def predict_env_dynamics(self, target_nodes, node_options, top_k):
        scene_dynamics = {
            obj_node: self.predict_object_dynamics(obj_node, node_options, top_k=top_k)
            for obj_node in target_nodes
        }
        return scene_dynamics

    @abstractmethod
    def predict_object_dynamics(self, object_node, node_options, top_k):
        pass

class RandomAgent(Agent):
    """
    Takes a purely random decision based on all options available.
    """
    def __init__(self, task_type):
        self.task_type = task_type

    def predict_location(self, target_node, node_options, ignore_nodes=[]):
        return random.choice(node_options)

    def predict_object_dynamics(self, object_node, node_options, top_k):
        rand_move_freq = random.randrange(0,10) / 10
        edge_prob_dict = {}
        while len(edge_prob_dict) != top_k:  # keep choosing random nodes till you reach k unique nodes 
            rand_node = random.choice(node_options)
            if rand_node not in edge_prob_dict:
                edge_prob_dict[rand_node] = random.randrange(0,10) / 10
        sum_rand_prob = sum(edge_prob_dict.values()) # since probabilities were randomly chosen, they may not sum up to 1.0
        if sum_rand_prob != 0:
            edge_prob_dict = {k:float(p_i/sum_rand_prob) for (k, p_i) in edge_prob_dict.items()} 
        return {'edge_prob': edge_prob_dict, 'move_freq': rand_move_freq}

class PriorsAgent(Agent):
    """
    Probabilistically chooses an action from available node options, weighted by priors.
    """

    def __init__(self, priors_graph, task_type):
        self.priors_graph = priors_graph
        self.task_type = task_type

    def predict_location(self, target_node, node_options, ignore_nodes=[]):
        edges_with_priors = self.priors_graph.get_edges_to(target_node.label, NodeType.FURNITURE)
        node_prob_dict = {}
        for edge in edges_with_priors:
            for node in node_options:
                if (not node.unique_id in ignore_nodes) and edge.node1.label == node.label:
                    node_prob_dict[node] = edge.prob
        try:
            choice = sample_from_dict(node_prob_dict)
        except:
            choice = random.choice(node_options)
        return choice

    def predict_object_dynamics(self, object_node, node_options, top_k):
        edges_with_priors = self.priors_graph.get_edges_to(object_node.label, NodeType.FURNITURE)
        edges_dict = {edge.node1: edge.prob for edge in edges_with_priors}
        all_prob_sum = sum(list(edges_dict.values()))
        if all_prob_sum != 0:
            edges_dict = {node: float(prob/all_prob_sum) for (node, prob) in edges_dict.items()}
        else:
            edges_dict = {}
        scene_edges_dict = {}
        for (node,prob) in edges_dict.items():
            scene_grp_node = self.lookup_in_scene_graph(node_options, node)
            if not scene_grp_node is None:
                scene_edges_dict[scene_grp_node] = prob

        edge_prob = dict(sorted(scene_edges_dict.items(), key=lambda x: x[1], reverse=True)[:top_k])
        return {'edge_prob': edge_prob, 'move_freq': None}

    def lookup_in_scene_graph(self, node_options, node):

        for node_i in node_options:
            if node_i.category == node.category:
                return node_i
        return None

class MemorizationAgent(RandomAgent, PriorsAgent):
    """
    Returns where the object was last seen. 
    If it has never seen the object before, uses a random choise (could be informed by priors).
    """

    def __init__(self, priors_graph, task_type, use_priors=False, observation_prob = 1.0, use_labels = True):
        RandomAgent.__init__(self, task_type)
        PriorsAgent.__init__(self, priors_graph, task_type)
        self.memory = {}
        self.observation_prob = observation_prob
        self.steps_since_seen = {}
        self.total_steps_elapsed = 0
        self.use_priors = use_priors
        self.use_labels = use_labels

    def transition_to_new_scene(self, scene):
        self.memory = {}

    def step(self):
        self.total_steps_elapsed += 1

    def predict_location(self, target_node, node_options, ignore_nodes=[]):
        if target_node.description in self.memory and (not self.memory[target_node.description].unique_id in ignore_nodes):
            return self.memory[target_node.description]
        elif target_node.label in self.memory and self.use_labels and (not self.memory[target_node.label].unique_id in ignore_nodes):
            return self.memory[target_node.label]
        elif self.use_priors:
            return PriorsAgent.predict_location(self, target_node, node_options, ignore_nodes)
        else:
            return RandomAgent.predict_location(self, target_node, node_options, ignore_nodes)
    
    def receive_predict_location_obs(self, node):
        for description in self.steps_since_seen:
            self.steps_since_seen[description]+=1
        for object_node in node.get_children_nodes():
            if random.random() > self.observation_prob:
                continue
            self.memory[object_node.description] = node
            #self.memory[object_node.label] = node
            if self.use_labels:
                self.steps_since_seen[object_node.description] = 0
                self.steps_since_seen[object_node.label] = 0

    def predict_object_dynamics(self, object_node, node_options, top_k):
        # Memorization remembers last location and steps since last seen for each obj
        # Top K Locations - only last known location with full probability. Cant do any better
        # Move freq - 
        # If steps since seen is large, low move probability
        # If steps since seen is n, and the current episode has had N steps total,
        # 1 - n/N --> close to 0: low prob of moving. Close to 1: high move prob
        if object_node.description in self.memory:
            edge_prob = {self.memory[object_node.description]:1.0}
            local_move_freq = 1 - (self.steps_since_seen[object_node.description] / self.total_steps_elapsed)
            return {'edge_prob': edge_prob, 'move_freq': local_move_freq}
        elif self.use_priors:
            return PriorsAgent.predict_object_dynamics(self, object_node, node_options, top_k)
        else:
            return RandomAgent.predict_object_dynamics(self, object_node, node_options, top_k)

class FrequentistAgent(RandomAgent, PriorsAgent):
    def __init__(self, priors_graph, task_type, use_priors=False, observation_prob=1.0):
        self.priors_graph = priors_graph
        self.task_type = task_type
        self.object_times_seen_at = {}
        self.object_times_seen = {}
        self.use_priors = use_priors
        self.observation_prob = observation_prob

    def transition_to_new_scene(self, scene):
        self.object_times_seen_at = {}
        self.object_times_seen = {}

    def receive_predict_location_obs(self, node):
        for object_node in node.get_children_nodes():
            if random.random() > self.observation_prob:
                continue
            object_desc = object_node.description
            if object_desc not in self.object_times_seen:
                self.object_times_seen[object_desc] = 0
                self.object_times_seen_at[object_desc] = {}
            self.object_times_seen[object_desc]+=1
            if node not in self.object_times_seen_at[object_desc]:
                self.object_times_seen_at[object_desc][node] = 0
            self.object_times_seen_at[object_desc][node]+=1

    def get_location_probs(self, object_node):
        node_probs = {}
        for furniture_node in self.object_times_seen_at[object_node.description]:
            times_seen_at_node = float(self.object_times_seen_at[object_node.description][furniture_node])
            times_seen = self.object_times_seen[object_node.description]
            node_probs[furniture_node] = times_seen_at_node/times_seen
        return node_probs

    def predict_location(self, target_node, node_options, ignore_nodes=[]):
        if target_node.description in self.object_times_seen_at:
            node_probs = self.get_location_probs(target_node)
            filtered_node_probs = {node:prob for (node,prob) in node_probs.items() if not node.unique_id in ignore_nodes}
            if len(filtered_node_probs) == 0:
                return RandomAgent.predict_location(self, target_node, node_options, ignore_nodes)
            return sample_from_dict(filtered_node_probs)
        elif self.use_priors:
            return PriorsAgent.predict_location(self, target_node, node_options, ignore_nodes)
        else:
            return RandomAgent.predict_location(self, target_node, node_options, ignore_nodes)

    def predict_object_dynamics(self, object_node, node_options, top_k):
        # Edge prob
        # frequentist memory knows all the locations that the object has been in last.
        # Does not know with what prob it can move between them. All equal probability.

        # Move freq
        # Get len(count memory) for all objs. The longer the length, the higher move freq.
        # Post processed in predict_env_dynamics so that move_freq is between 0 and 1.0
        if object_node.description in self.frequentist_memory:
            node_probs = self.get_location_probs(object_node)
            #TODO actually filter down to k
            move_freq = len(self.object_times_seen_at[object_node.desceription])/float(sum(self.object_times_seen_at[object_node.description]))
            return {'edge_prob': node_probs, 'move_freq': move_freq}
        elif self.use_priors:
            return PriorsAgent.predict_object_dynamics(self, object_node, node_options, top_k)
        else: 
            return RandomAgent.predict_object_dynamics(self, object_node, node_options, top_k)

class FrequentistWithMemorizationAgent(FrequentistAgent, MemorizationAgent):
    """
    Uses the Memorization Agent if the object was last seen less than 100 steps ago.
    Otherwise uses the frequentist Agent.
    """
    def __init__(self, priors_graph, task_type, use_priors=False):
        FrequentistAgent.__init__(self, priors_graph, task_type, use_priors)
        MemorizationAgent.__init__(self, priors_graph, task_type, use_priors)

    def transition_to_new_scene(self, scene):
        FrequentistAgent.transition_to_new_scene(self, scene)
        MemorizationAgent.transition_to_new_scene(self, scene)

    def predict_location(self, target_node, node_options, ignore_nodes=[]):
        if target_node not in self.steps_since_seen:
            return RandomAgent.predict_location(self, target_node, node_options, ignore_nodes)
        elif self.steps_since_seen[target_node] < 100:
            return MemorizationAgent.predict_location(self, target_node, node_options, ignore_nodes)
        else:
            return FrequentistAgent.predict_location(self, target_node, node_options, ignore_nodes)

    def receive_predict_location_obs(self, node):
        FrequentistAgent.receive_predict_location_obs(self, node)
        MemorizationAgent.receive_predict_location_obs(self, node)

    def predict_object_dynamics(self, object_node, node_options, top_k):
        if object_node not in self.steps_since_seen:
            return RandomAgent.predict_object_dynamics(self, object_node, top_k)
        elif self.steps_since_seen[object_node] < 100:
            return MemorizationAgent.predict_object_dynamics(self, object_node, top_k)
        else:
            return FrequentistAgent.predict_object_dynamics(self, object_node, top_k)

class BayesianAgent(PriorsAgent):

    class BetaDistribution:
        def __init__(self, a = 1, b = 1):
            self.a = a
            self.b = b
            self.times_observed = 0
            self.times_true = 0

        def get_pdf(self):
            x = np.linspace(0, 1, 100)
            fx = beta.pdf(x, self.a, self.b)
            return fx

        def get_mean(self):
            return beta.mean(self.a, self.b)
            
        def update_with_observation(self, obs_is_true):
            self.times_observed+=1
            if obs_is_true:
                self.times_true+=1
                self.a+=1
            else:
                self.b+=1 

        @staticmethod
        def compute_starting_a_b(prior_prob, var):
            mu = prior_prob
            alpha = mu**2 * ((1 - mu) / var - 1 / mu)
            beta = alpha * (1 / mu - 1)
            return alpha, beta

    def __init__(self, priors_graph, task_type, observation_prob=1.0, prior_var=0.25):
        super().__init__(priors_graph, task_type)
        self.priors_graph = priors_graph
        self.observation_prob = observation_prob
        self.prior_var = prior_var
        self.beta_distributions = {} 

    def transition_to_new_scene(self, scene):
        self.beta_distributions = {} 

    def init_dist_for_edge(self, object_node, furniture_node):
        prior_furniture_node = self.priors_graph.get_node_with_category(furniture_node.category)
        prior_object_nodes = self.priors_graph.get_nodes_with_label(object_node.label)

        priors_edge = None
        for prior_object_node in prior_object_nodes:
            possible_priors_edge = prior_furniture_node.get_edge_to(prior_object_node)
            if possible_priors_edge is not None:
                priors_edge = possible_priors_edge
                break

        if priors_edge is None or priors_edge.prob == 0:
            self.beta_distributions[object_node.description][furniture_node] = None
        else:
            prior_prob = min(priors_edge.prob,0.99) #one case of 1.0, TODO look into
            var = min(prior_prob*(1-prior_prob)*0.9, self.prior_var)
            a, b = self.BetaDistribution.compute_starting_a_b(prior_prob, var)
            self.beta_distributions[object_node.description][furniture_node] = self.BetaDistribution(a,b)

    def receive_predict_location_obs(self, observation_node):

        for object_node in observation_node.get_children_nodes():
            if random.random() > self.observation_prob:
                continue
            if object_node.description not in self.beta_distributions:
                self.beta_distributions[object_node.description] = {}
            beta_dists = self.beta_distributions[object_node.description]
            if observation_node not in beta_dists:
                self.init_dist_for_edge(object_node, observation_node)
            for furniture_node in beta_dists:
                beta_dist = beta_dists[furniture_node]
                if beta_dist is not None:
                    beta_dist.update_with_observation(furniture_node == observation_node)

    def predict_location(self, target_node, node_options, ignore_nodes=[]):
        if target_node.description not in self.beta_distributions:
            self.beta_distributions[target_node.description] = {}
        target_dists = self.beta_distributions[target_node.description]

        node_prob_dict = {}
        for furniture_node in node_options:
            if furniture_node not in target_dists:
                self.init_dist_for_edge(target_node, furniture_node)
            beta_dist = target_dists[furniture_node]
            if beta_dist is not None:
                node_prob_dict[furniture_node] = beta_dist.get_mean()

        if len(node_prob_dict) == 0:
            # rare bug, TODO figure out cause
            return random.choice(node_options)

        choice = sample_from_dict(node_prob_dict)
        return choice

class SGMAgent(Agent):
    def __init__(self,
                 priors_graph,
                 task_type,
                 node_featurizer=None,
                 edge_featurizer=None,
                 model=None,
                 constant_prior_probs=True,
                 prior_prob_threshold=0.2,
                 observation_prob=1.0,
                 linear_schedule_num_steps=50,
                 memorization_baseline=False,
                 use_edge_weights=True,
                 use_priors=True,
                 use_model=True,
                 add_num_nodes=True,
                 add_num_edges=True,
                 reverse_edges=True):#baseline with no prior edges
        self.priors_graph = priors_graph
        self.task_type = task_type
        self.node_featurizer = node_featurizer
        self.edge_featurizer = edge_featurizer
        self.sgm_graph = SGM()
        self.SGM_to_scene_node_map = {}
        self.constant_prior_probs = constant_prior_probs #whether to update probabilities
        self.prior_prob_threshold = prior_prob_threshold
        self.observation_prob = observation_prob
        self.linear_schedule_num_steps = linear_schedule_num_steps
        self.memorization_baseline = memorization_baseline
        self.model = model
        self.use_edge_weights = use_edge_weights
        self.use_priors = use_priors
        self.use_model = use_model
        self.add_num_nodes = add_num_nodes
        self.add_num_edges = add_num_edges
        self.reverse_edges = reverse_edges
        self.to_undirected = ToUndirected()

    def receive_predict_location_obs(self, observation_node):
        """
        Means you get to know about all the children of this node from
        the scene graph, and copy to SGM as needed
        """
        # can see all the children of this node
        node_in_sgm = self.sgm_graph.get_node_with_unique_id(observation_node.unique_id)
        if node_in_sgm is None:
            raise ValueError('No node with id %s in SGM'%observation_node.unique_id)
        observed_edges = set()
        for edge in observation_node.get_edges_from_me():
            if random.random() > self.observation_prob:
                continue
            child_node = edge.node2
            sgm_child_node = self.sgm_graph.get_node_with_description(child_node.description)
            if sgm_child_node is None:
                # Object has not been observed or picked as goal before
                sgm_child_node = child_node.copy_to_sgm(from_observation=True)
                self.SGM_to_scene_node_map[sgm_child_node] = child_node
                self.sgm_graph.add_node(sgm_child_node)
                #if not self.memorization_baseline:
                #    self.add_hypothetical_edges(sgm_child_node, 
                #                            zero_edge_probs=not self.constant_prior_probs, 
                #                            observation_node=observation_node)
            else:
                self.remove_prior_hyp_edges(sgm_child_node)

            if self.memorization_baseline:
                sgm_child_node.reset_edges()

            observed_edge_in_sgm = None
            for sgm_edge in [c_edge for c_edge in node_in_sgm.edges if c_edge.node2 == sgm_child_node]:
                if edge.type == sgm_edge.type:
                    observed_edge_in_sgm = sgm_edge

            if observed_edge_in_sgm is None:
                # May not have edge if low probability in prior
                observed_edge_in_sgm = SGMEdge(node_in_sgm,
                                               sgm_child_node,
                                               edge.type,
                                               prob=self.prior_prob_threshold,
                                               from_priors=False)
            else:
                observed_edges.add(observed_edge_in_sgm)
                observed_edge_in_sgm.observe(True, set_prob_to_one = not self.constant_prior_probs)

            node_state_has_changed = observed_edge_in_sgm.time_since_state_change == 0
            node_in_sgm.observe(state_has_changed = False)
            sgm_child_node.observe(state_has_changed = node_state_has_changed)

        for sgm_edge in node_in_sgm.get_edges_from_me():
            if sgm_edge not in observed_edges:
                continue
            sgm_edge.observe(state=False, set_prob_to_one=False)

    def add_hypothetical_edges(self, node, zero_edge_probs=False, observation_node=None):
        if self.use_priors:
            self.add_prior_hyp_edges(node, zero_edge_probs, observation_node)
        else:
            self.add_non_prior_hyp_edges(node)

    def add_prior_hyp_edges(self, node, zero_edge_probs=False, observation_node=None):
        if node.from_observation:
            priors_nodes = self.priors_graph.get_nodes_with_category(node.category)
        else:
            priors_nodes = self.priors_graph.get_nodes_with_label(node.label)
        total_prob = 0.0
        new_edges = []
        for priors_node in priors_nodes:
            for priors_edge in priors_node.edges:

                from_nodes = self.sgm_graph.get_nodes_with_label(priors_edge.node1.label)

                for from_node in from_nodes:
                    if from_node.has_edge_to(node, priors_edge.type):
                        continue
                    if observation_node is not None and from_node.category != observation_node.category:
                        continue
                    prob = 0.0 if zero_edge_probs else priors_edge.prob
                    total_prob+=prob
                    new_edge = SGMEdge(from_node,
                                       node,
                                       priors_edge.type,
                                       prob=prob,
                                       from_priors=True)
                    new_edges.append(new_edge)

        if not zero_edge_probs and total_prob!=0: #normalize
            for edge in new_edges:
               edge.prob/=total_prob

        return new_edges

    def remove_prior_hyp_edges(self, node):
        edges_to_remove = [edge for edge in node.edges if edge.from_priors]
        for edge in edges_to_remove:
            edge.remove_from_nodes()

    def add_non_prior_hyp_edges(self, node):
        # add nodes from SGM, or random

        sgm_nodes = [sgm_node for sgm_node in self.sgm_graph.get_nodes_with_label(node.label) if sgm_node!=node]

        possible_edges = []
        for sgm_node in sgm_nodes:
            possible_edges += [edge for edge in sgm_node.edges]
        for possible_edge in possible_edges:
            if possible_edge.node1.has_edge_to(node, possible_edge.type):
                continue
            SGMEdge(possible_edge.node1, node, possible_edge.type, prob=possible_edge.prob)

        if len(sgm_nodes) == 0:
            possible_parent_nodes = self.sgm_graph.get_nodes_with_type(NodeType.FURNITURE)
            for possible_parent_node in random.choices(possible_parent_nodes, k=5):
                if not possible_parent_node.has_edge_to(node, EdgeType.CONTAINS):
                    SGMEdge(possible_parent_node,
                            node,
                            EdgeType.CONTAINS,
                            prob=0)
                if not possible_parent_node.has_edge_to(node, EdgeType.UNDER):
                    SGMEdge(possible_parent_node,
                            node,
                            EdgeType.UNDER,
                            prob=0)

    def transition_to_new_scene(self, scene):
        self.sgm_graph = SGM()
        self.SGM_to_scene_node_map = {}
        self.scene_furniture_nodes = scene.scene_graph.get_nodes(node_type = NodeType.FURNITURE)
        #assume we know all non-object nodes initially
        for node in scene.scene_graph.nodes:
            if node.type != NodeType.OBJECT:
                sgm_node = node.copy_to_sgm(from_observation=True)
                self.SGM_to_scene_node_map[sgm_node] = node
                self.sgm_graph.add_node(sgm_node)
        for sgm_node in self.sgm_graph.nodes:
            if sgm_node.type != NodeType.FURNITURE:
                sg_node = self.SGM_to_scene_node_map[sgm_node]
                for edge in sg_node.get_edges_from_me():
                    sgm_end_node = self.sgm_graph.get_node_with_unique_id(edge.node2.unique_id)
                    SGMEdge(sgm_node, sgm_end_node, edge.type, prob=0.0)

    def step(self):
        """
        Take time passing into account for the SGM graph
        """
        self.sgm_graph.step()
        """
        remove_nodes = [node for node in self.sgm_graph.nodes if node.type == NodeType.OBJECT and node.time_since_observed > 50]
        for node in remove_nodes:
            self.sgm_graph.remove_node(node)
        """
        if not self.constant_prior_probs:
            # Also account for time passing
            for object_node in self.sgm_graph.get_nodes(node_type=NodeType.OBJECT):
                if object_node.time_since_observed < 0 or object_node.time_since_observed > self.linear_schedule_num_steps:
                    continue

                has_edges_from = {edge.node1.category for edge in object_node.edges}
                priors_node = self.priors_graph.get_node_with_category(object_node.category)
                #Make a dict to make calculation easier
                label_to_prior_prob = {}
                for edge in priors_node.edges:
                    if edge.node1.unique_id in has_edges_from and edge.node2.label == object_node.label:
                        label_to_prior_prob[edge.node1.category] = edge.prob

                total_probs = 0.0
                # Compute new unnormalized probabilities
                for edge in object_node.edges:
                    if edge.node1.category not in label_to_prior_prob:
                        prior_prob = 0
                    else:
                        prior_prob = label_to_prior_prob[edge.node1.category]
                    from_prior = (edge.time_since_observed / self.linear_schedule_num_steps) * prior_prob / len(object_node.edges)
                    from_sgm = (self.linear_schedule_num_steps - edge.time_since_observed) / self.linear_schedule_num_steps * 1.0
                    new_prob = from_prior + from_sgm #linear interpolation from 1.0 back to prior
                    new_prob = max(new_prob, 1.0)
                    edge.prob = new_prob
                    total_probs+=new_prob

                if total_probs < 0.01:
                    continue # no need to normalize (???)

                if not self.memorization_baseline:
                    for edge in object_node.edges:
                        edge.prob/=total_probs

    def make_predict_location_inputs(self, query_node, ignore_nodes=[]):
        sgm_furniture_node_options = self.sgm_graph.get_nodes(node_type = NodeType.FURNITURE)
        option_nums = [self.sgm_graph.get_node_num(node) for node in sgm_furniture_node_options]
        edges = []
        edges_to_eval = []
        edge_features = []
        choice_options = []
        target_node_num = self.sgm_graph.get_node_num(query_node)
        for i, node in enumerate(sgm_furniture_node_options):
            if node.unique_id in ignore_nodes:
                continue
            edges_to_target = node.get_edges_to(query_node)
            for edge in edges_to_target:
                if edge.node1.type != NodeType.FURNITURE:
                    continue
                edges.append(edge)
                node_edge_features = self.edge_featurizer.featurize_to_vec(edge)
                if self.add_num_nodes:
                    node_edge_features = np.array(list(node_edge_features) + [len(self.sgm_graph.nodes)/250.0])
                if self.add_num_edges:
                    node_edge_features = np.array(list(node_edge_features) + [len(self.sgm_graph.get_edges())/500.0])
                edge_features.append(node_edge_features)
                if self.reverse_edges:
                    edges_to_eval.append(np.array([target_node_num, option_nums[i]]))
                else:
                    edges_to_eval.append(np.array([option_nums[i], target_node_num]))
                choice_options.append(node)
        return choice_options, edges_to_eval, edge_features, edges

    def make_predict_locations_inputs_heterogeneous(self, query_node, data, ignore_nodes=[]):
        target_node_id_in_pyg = data['object'].mapping[query_node.unique_id]

        subgraph = data.subgraph({
            'object': torch.tensor([target_node_id_in_pyg], dtype=torch.long),
            'furniture': torch.arange(data['furniture'].x.shape[0])
        })


        node_candidates = {}
        edge_candidates = {}
        edges_to_eval_dict = {}
        edge_features_dict = {}

        # This also needs to be per category...
        for key, value in subgraph.edge_items(): 
            edges_to_eval_dict[key] = torch.unsqueeze(value['edge_index'].transpose(0, 1), dim=0)
            edge_features_dict[key] = torch.unsqueeze(value['edge_attr'], dim=0)
            edge_candidates[key] = []
            if key.index('furniture') == 0:
                idx = 0
            else:
                idx = 1

            nodes = []
            edges = []
            node_ids = subgraph['furniture'].id[value.edge_index[idx].cpu()]
            if type(node_ids) == np.str_:
                node_ids = np.array([node_ids])
            
            for node_id in node_ids:
                node = self.sgm_graph.get_node_with_unique_id(node_id)
                if node is None:
                    continue
                nodes.append(node)
                for to_edge in node.get_edges_to(query_node):
                    if self.reverse_edges:
                        edge_type = RECIPROCAL_EDGE_TYPES[to_edge.type].value
                    else:
                        edge_type = to_edge.type.value
                    if edge_type == key[1]:
                        edges.append(to_edge)
            
            node_candidates[key] = nodes
            edge_candidates[key] = edges

        return node_candidates, edge_candidates, edges_to_eval_dict, edge_features_dict


    def prepare_predict_location(self):
        for edge in self.sgm_graph.get_edges():
            edge.last_sgm_prob = 0
            edge.sgm_model = self.model.get_model_type()

        for node in self.sgm_graph.nodes:
            node.normalize_edges_to_me()

    def compute_model_outputs(self, edges_to_eval, edge_features, edge_key=None):
        edges_to_eval = torch.unsqueeze(torch.tensor(np.stack(edges_to_eval), dtype=torch.long).cuda(),dim=0)
        edge_features = torch.unsqueeze(torch.tensor(np.stack(edge_features), dtype=torch.float).cuda(), dim=0)

        if self.reverse_edges:
            graph_metadata = REVERSED_GRAPH_METADATA
        else:
            graph_metadata = GRAPH_METADATA

        data = graph_to_pyg(self.sgm_graph, self.node_featurizer, self.edge_featurizer, 
                            heterogenous=self.model.is_heterogenous(), 
                            graph_metadata=graph_metadata,
                            include_labels=False, add_mapping=True)

        if not self.use_edge_weights:
            data.edge_weights = None
        data = data.cuda()
        out = compute_output(self.model, data, edges_to_eval, edge_features, edge_key)
        out = torch.squeeze(out).cpu().tolist()
        if type(out) is float:
            out = [out]
        return out

    def get_query_node(self, target_node):
        if len(self.sgm_graph.get_nodes(description=target_node.description)) == 0:
            sgm_goal_node = target_node.copy_to_sgm(from_observation=False)
            self.sgm_graph.add_node(sgm_goal_node)
        query_node = self.sgm_graph.get_nodes(description=target_node.description)[0]
        return query_node

    def predict_location_heterogeneous(self, target_node, node_options, return_probs, ignore_nodes):
        assert self.model

        sgm_query_node = self.get_query_node(target_node)
        self.prepare_predict_location()

        if self.reverse_edges:
            graph_metadata = REVERSED_GRAPH_METADATA
        else:
            graph_metadata = GRAPH_METADATA

        self.add_hypothetical_edges(sgm_query_node)

        data = graph_to_pyg(
            self.sgm_graph,
            self.node_featurizer,
            self.edge_featurizer, 
            heterogenous=True,
            graph_metadata=graph_metadata,
            include_labels=False,
            add_mapping = True,
        )

        # TODO: fix
        if ("furniture", "onTop", "furniture") in data.edge_types:
            del data[("furniture", "onTop", "furniture")]

        if ("furniture", "in", "furniture") in data.edge_types:
            del data[("furniture", "in", "furniture")]

        if not self.use_edge_weights:
            for key in data.edge_types: #type: ignore
                data[key].edge_weights = None #type: ignore

        data = data.cuda()

        # TODO: we may not need choice_options_dict
        node_candidates, edge_candidates, edges_to_eval_dict, edge_features_dict = self.make_predict_locations_inputs_heterogeneous(sgm_query_node, data, ignore_nodes)
        output_keys = {}
        for key in edges_to_eval_dict: #type:ignore
            out = compute_output(self.model, data, edges_to_eval_dict[key], edge_features_dict[key], key)
            out = torch.squeeze(out).cpu().tolist()
            if type(out) is float:
                out = [out]
            output_keys[key] = out

        # TODO: This part does not work
        for key in output_keys:
            edges = edge_candidates[key] # type: ignore
            for i in range(len(edges)):
                edges[i].last_sgm_prob = output_keys[key][i]

        # TODO: Does this get normalized if we assign by reference?
        sgm_query_node.normalize_sgm_probs()

        if return_probs:
            raise Exception("Return probs not supported")
            # node_prob_dict = {
            #     self.SGM_to_scene_node_map[node_choice]: pred_prob 
            #     for (node_choice, pred_prob) in zip(node_candidates[key], out)
            # }
            # return node_prob_dict
        out = np.concatenate(list(output_keys.values()))
        nodes = np.concatenate(list(node_candidates.values()))
        choice = nodes[np.argmax(out)]
        return self.SGM_to_scene_node_map[choice]

    def predict_location_homogeneous(self, target_node, node_options, return_probs, ignore_nodes):
        sgm_query_node = self.get_query_node(target_node)
        self.add_hypothetical_edges(sgm_query_node)
        choice_options, edges_to_eval, edge_features, edges = self.make_predict_location_inputs(sgm_query_node, ignore_nodes)
        self.prepare_predict_location()

        if len(edges_to_eval) == 0:
            return None

        if not self.use_model:
            out = [random.random() for i in range(len(edges_to_eval))]
        else:
            #choice = random.choices(list(range(len(out))), k=1, weights=out)[0]
            out = self.compute_model_outputs(edges_to_eval, edge_features)
            #confidence = max(out)/np.mean(out) - 1 # 0 is random, 1 or more is super confident
            #print(confidence)

        for i in range(len(edges)):
            edges[i].last_sgm_prob = out[i]

        sgm_query_node.normalize_sgm_probs()
        self.remove_prior_hyp_edges(sgm_query_node)
        # I think we should get rid of this?
        # sgm_query_node.normalize_sgm_probs()

        if return_probs:
            node_prob_dict = {
                self.SGM_to_scene_node_map[node_choice]: pred_prob 
                for (node_choice, pred_prob) in zip(choice_options, out)
            }
            return node_prob_dict
        choice = choice_options[out.index(max(out))]
        return self.SGM_to_scene_node_map[choice]

    def predict_location(self, object_node, node_options, return_probs=False, ignore_nodes=[]):
        if self.use_model and self.model.is_heterogenous(): #type: ignore
            prob_preds = self.predict_location_heterogeneous(object_node, node_options, return_probs, ignore_nodes)
        else:
            prob_preds = self.predict_location_homogeneous(object_node, node_options, return_probs, ignore_nodes)
        return prob_preds

    def get_sgm_graph(self):
        return self.sgm_agent.sgm_graph

    def predict_object_dynamics(self, object_node, node_options, top_k):
        prob_preds = self.predict_location(object_node, node_options, return_probs=True)
        edge_prob = dict(sorted(prob_preds.items(), key=lambda x: x[1], reverse=True)[:top_k])
        return {"edge_prob": edge_prob}

class SGMDataCollectionAgent(Agent):
    """
    We may want to run one policy (e.g. random) while storing observations
    via a SGM agent. This class makes this elegant.
    """
    def __init__(self, policy_agent, sgm_agent):
        self.policy_agent = policy_agent
        self.sgm_agent = sgm_agent

    def receive_observation(self, observation):
        self.policy_agent.receive_observation(observation)
        self.sgm_agent.receive_observation(observation)

    def transition_to_new_scene(self, scene):
        self.policy_agent.transition_to_new_scene(scene)
        self.sgm_agent.transition_to_new_scene(scene)

    def step(self):
        self.policy_agent.step()
        self.sgm_agent.step()

    def make_predictions(self, task_type, scene, query, top_k=3, ignore_nodes_to_pred=[]):
        # this makes sure the SGM graph adds new nodes for the query 
        for edge in self.sgm_agent.sgm_graph.get_edges():
            edge.is_query_edge = False
        if type(query) is Node:
            query_sgm_nodes = [self.sgm_agent.get_query_node(query)]
        else:
            query_sgm_nodes = []
            for q in query:
                query_sgm_nodes+=[self.sgm_agent.get_query_node(q)]
        for node in query_sgm_nodes:
            self.sgm_agent.add_hypothetical_edges(node)
            for edge in node.edges:
                edge.is_query_edge = True
        self.query_sgm_nodes = query_sgm_nodes

        return self.policy_agent.make_predictions(task_type, scene, query, top_k=top_k, ignore_nodes_to_pred=ignore_nodes_to_pred)

    def mark_true_edges(self, scene_graph):
        """
        Set a variable true for SGM edges that are actually present
        in the scene graph. Used for supervised training.
        """
        for sgm_edge in self.sgm_agent.sgm_graph.get_edges():
            sgm_edge.currently_true = False
        for sgm_edge in self.sgm_agent.sgm_graph.get_edges():
            sg_node1 = scene_graph.get_node_with_unique_id(sgm_edge.node1.unique_id)
            sg_nodes2 = scene_graph.get_nodes_with_description(sgm_edge.node2.description)
            for sg_node2 in sg_nodes2:
                sgm_edge.currently_true = sgm_edge.currently_true or sg_node1.has_edges_to(sg_node2)

    def remove_hyp_query_edges(self):
        for node in self.query_sgm_nodes:
            self.sgm_agent.remove_prior_hyp_edges(node)

class UpperBoundAgent(MemorizationAgent):

    def __init__(self, priors_graph, scene_sampler, scene_evolver, observation_prob, task_type, dont_predict_prior_node=False, oracle=False):
        super().__init__(
            priors_graph=priors_graph,
            task_type=task_type, 
            use_priors=False,
            observation_prob=observation_prob
        )
        if oracle:
            self.use_priors = True
        self.parent_memory = {}
        self.scene_sampler = scene_sampler
        self.scene_evolver = scene_evolver
        self.dont_predict_prior_node = dont_predict_prior_node
        self.oracle = oracle

    def transition_to_new_scene(self, scene):
        self.memory = {}
        if self.oracle:
            self.prior_graph = self.scene_sampler.current_priors_graph
            for node in scene.scene_graph.get_nodes_with_type(NodeType.FURNITURE):
                for object_node in node.get_children_nodes():
                    self.memory[object_node.description] = node
                    self.steps_since_seen[object_node.description] = 0

    def predict_location(self, target_node, node_options, ignore_nodes=[]):
        scene_graph = self.scene_evolver.scene.scene_graph
        if target_node.description in self.steps_since_seen:
            steps_passed = self.steps_since_seen[target_node.description]
            object_1step_move_prob = scene_graph.get_object_node_move_probs()[target_node]
            object_has_not_moved_prob = (1.0 - object_1step_move_prob)**(steps_passed*self.scene_evolver.object_moves_per_step)
        else:
            object_has_not_moved_prob = 0.0

        if object_has_not_moved_prob > 0.5 and target_node.description in self.memory:
            return super().predict_location(target_node, node_options, ignore_nodes)
        else:
            ignore_graph_nodes = [self.scene_evolver.scene.scene_graph.get_node_with_unique_id(node_id) for node_id in ignore_nodes]
            if self.dont_predict_prior_node and target_node.description in self.memory:
                if target_node.description in self.memory:
                    prior_node = self.memory[target_node.description]
                else:
                    prior_node = super(PriorsAgent,self).predict_location(target_node, node_options, ignore_nodes)
                option_probs = self.scene_evolver.get_move_target_probs(target_node, ignore_graph_nodes + [prior_node])
            else:
                option_probs = self.scene_evolver.get_move_target_probs(target_node, ignore_graph_nodes)
            if len(option_probs) != 0:
                node_choice, edge_choice = sample_from_dict(option_probs)
            else:
                node_choice = random.choice(node_options) 
            return node_choice

    def predict_object_dynamics(self, object_node, node_options, top_k):
        scene_graph = self.scene_evolver.scene.scene_graph
        if object_node.description in self.steps_since_seen:
            move_freq = scene_graph.get_object_node_move_probs()[object_node]
            option_probs = self.scene_evolver.get_move_target_probs(object_node)
            edge_prob = dict(
                    sorted(option_probs.items(), key=lambda item: item[1], reverse=True)[:top_k]
                )
            edge_prob_dict = {k[0]:v for (k,v) in edge_prob.items()}
            return {'edge_prob': edge_prob_dict, 'move_freq': move_freq}
        else:
            return MemorizationAgent.predict_object_dynamics(self, object_node, node_options, top_k)

def make_agent(cfg, 
               agent_type, 
               task_type, 
               node_featurizer = None, 
               edge_featurizer = None, 
               scene_sampler=None, 
               scene_evolver=None, 
               for_data_collection=False, 
               model_cfg=None,
               load_model=True):
    if 'ok_priors' in agent_type:
        cfg.agent_priors_type = 'coarse'
    if 'best_priors' in agent_type:
        cfg.agent_priors_type = 'detailed'

    if 'sgm' in agent_type and agent_type!='sgm':
        if 'gcn' in agent_type and 'hgcn' not in agent_type:
            model_cfg.model_type = 'gcn'
            model_cfg.model_name = 'gcn'
        elif 'hgt' in agent_type:
            model_cfg.model_type = 'hgt'
            model_cfg.model_name = 'hgt'
        elif 'heat' in agent_type:
            model_cfg.model_type = 'heat'
            model_cfg.model_name = 'heat'
        elif 'hgnn' in agent_type:
            model_cfg.model_type = 'hgnn'
        elif 'hgcn' in agent_type:
            model_cfg.model_type = 'hgcn'
            model_cfg.model_name = 'hgcn'
        elif 'han' in agent_type:
            model_cfg.model_type = 'han'
            model_cfg.model_name = 'han'
        elif 'mlp' in agent_type:
            model_cfg.model_type = 'mlp'
            model_cfg.model_name = 'mlp'
        agent_type = 'sgm'

    sgm_agent = None
    priors_graph = load_priors_graph(cfg.agent_priors_type)
    for node in priors_graph.nodes:
        node.normalize_edges_to_me()

    if agent_type == 'random':
        agent = RandomAgent(task_type)
    elif agent_type == 'priors':
        agent = PriorsAgent(priors_graph, task_type)
    elif agent_type == 'memorization':
        agent = MemorizationAgent(priors_graph, task_type, cfg.memorization_use_priors, cfg.observation_prob)
    elif agent_type == 'upper_bound':
        agent = UpperBoundAgent(priors_graph, scene_sampler, scene_evolver, cfg.observation_prob, task_type)
    elif agent_type == 'oracle_upper_bound':
        agent = UpperBoundAgent(priors_graph, scene_sampler, scene_evolver, cfg.observation_prob, task_type, oracle=True)
    elif agent_type == 'frequentist':
        agent = FrequentistAgent(priors_graph, task_type, cfg.frequentist_use_priors, cfg.observation_prob)
    elif agent_type == 'frequentist_with_memorization':
        agent = FrequentistWithMemorizationAgent(priors_graph, task_type, use_priors=cfg.frequentist_use_priors)
    elif agent_type == 'bayesian':
        agent = BayesianAgent(priors_graph, task_type, cfg.observation_prob, cfg.bayesian_prior_var)
    elif agent_type == 'sgm':
        if load_model:
            model = make_model(model_cfg, node_featurizer, edge_featurizer, load_model=True)
            model.eval()
        else:
            model = None
        agent = SGMAgent(
                priors_graph = priors_graph,
                task_type = task_type, 
                node_featurizer = node_featurizer,
                edge_featurizer = edge_featurizer,
                model = model,
                constant_prior_probs = cfg.constant_prior_probs,
                linear_schedule_num_steps=cfg.linear_schedule_num_steps,
                observation_prob=cfg.observation_prob,
                memorization_baseline = cfg.sgm_memorization_baseline,
                use_priors = cfg.sgm_use_priors,
                use_model = cfg.sgm_use_model,
                add_num_nodes = model_cfg.add_num_nodes,
                add_num_edges = model_cfg.add_num_edges,
                reverse_edges = model_cfg.reversed_edges)
        sgm_agent = agent
    else:
        raise ValueError('%s is not a valid agent type'%agent_type)

    if for_data_collection:
        sgm_agent = make_agent(cfg, 'sgm', task_type, model_cfg = model_cfg, load_model = False)
        agent = SGMDataCollectionAgent(agent, sgm_agent)

    return agent
