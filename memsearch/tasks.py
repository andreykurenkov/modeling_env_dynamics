import gym
import random
from memsearch.util import sample_from_dict
from memsearch.graphs import NodeType
from enum import Enum
from abc import abstractmethod

from memsearch.scene import get_cost_between_nodes 


class TaskType(Enum):
    PREDICT_LOC = "predict_location"
    PREDICT_ENV_DYNAMICS = "predict_env_dynamics"
    FIND_OBJECT = "find_object"

class Task(gym.Env):
    """A goal-based environment node choice env.
    Choose the correct parent node for a given goal node.
    Observation and action space implemented with integers corresponding
    to nodes in the scene graph.
    """

    def __init__(self,
                 scene_sampler,
                 scene_evolver,
                 eps_per_scene=250,
                 scene_obs_percent=0.1,
                 pct_query_moved_object=0.5):
        self.scene_sampler = scene_sampler
        self.scene_evolver = scene_evolver
        self.eps_per_scene = eps_per_scene
        self.pct_query_moved_object = pct_query_moved_object
        """
        self.action_space = spaces.Discrete(max_nodes)
        self.observation_space = spaces.Dict(dict(
            node_features=spaces.Box(low=0, high=1, shape=(max_nodes,features_per_node), dtype=np.float32),
            edges=spaces.Box(low=0, high=1, shape=(2,max_edges), dtype=np.float32),
            num_nodes=spaces.Discrete(max_nodes),
            num_edges=spaces.Discrete(max_edges),
            achieved_goal=spaces.Box(low=0, high=1, shape=(goal_embedding_size,), dtype=np.int32),
            desired_goal=spaces.Box(low=0, high=1, shape=(goal_embedding_size,), dtype=np.int32),
            action_mask=spaces.Box(0, 1, shape=(max_avail_actions, )),
        ))
        """
        self.current_ep = 0
        self.goal_node = None
        self.scene_obs_percent = scene_obs_percent
        self.scene = None

    def pick_query(self, moved_objects, num_to_choose=1):
        all_object_nodes = self.scene.scene_graph.get_nodes_with_type(NodeType.OBJECT)
        object_move_probs = self.scene.scene_graph.get_object_node_move_probs()
        if num_to_choose == 1:
            if random.random() > self.pct_query_moved_object:
                query = sample_from_dict(object_move_probs)
            else:
                sample_from = all_object_nodes if len(moved_objects) == 0 else moved_objects
                query = random.choice(sample_from)
            self.goal_node = query
        else:
            query = []
            while len(query) < num_to_choose:
                if random.random() > self.pct_query_moved_object:
                    choice = sample_from_dict(object_move_probs)
                else:
                    sample_from = all_object_nodes if len(moved_objects) == 0 else moved_objects
                    choice = random.choice(sample_from)
                    if query in moved_objects:
                        moved_objects.remove(query)
                if choice in query:
                    continue
                query.append(choice)
        return query

    def reset(self):
        if self.current_ep==0 or self.current_ep % self.eps_per_scene == 0:
            self.scene = self.scene_sampler.sample()
            self.scene_evolver.set_new_scene(self.scene)
        moved_objects = self.scene_evolver.evolve()
        for edge in self.scene.scene_graph.get_edges():
            edge.age+=1
        for node in moved_objects:
            node.times_moved+=1
        self.current_ep+=1
        return moved_objects

    @abstractmethod
    def get_task_type(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (Node): the node to explore
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError()

    def get_random_furniture_node(self):
        all_furniture_nodes = self.scene.scene_graph.get_nodes(NodeType.FURNITURE)
        return random.choice(all_furniture_nodes)

class PredictLocationsTask(Task):
    def __init__(self, 
                scene_sampler, 
                scene_evolver, 
                eps_per_scene=250, 
                scene_obs_percent=0.1, 
                num_objs_to_predict=1):
        super().__init__(scene_sampler, 
                        scene_evolver, 
                        eps_per_scene, 
                        scene_obs_percent)
        assert num_objs_to_predict >= 1, "Cannot predict the location of fewer than one object."
        self.num_objs_to_predict = num_objs_to_predict

    def get_task_type(self):
        return TaskType.PREDICT_LOC

    def reset(self):
        moved_objects = super().reset()
        query = self.pick_query(moved_objects, self.num_objs_to_predict)
        return query

    def step(self, action):
        reward = 0
        if not isinstance(action, dict):
            action = {self.goal_node.description: action}
        for (goal_node_description, predicted_parent_node) in action.items():
            for child in predicted_parent_node.get_children_nodes():
                if child.description == goal_node_description:
                    reward += 1
        reward = float(reward / len(action))
        done = True
        info = {}
        obs = list(action.values())[0] # gets to observe the node
        return obs, reward, done, info

class FindObjTask(Task):

    def reset(self):
        moved_objects = super().reset()
        query = self.pick_query(moved_objects, 1)
        return query 

    def step(self, action, agent=None, current_scene=None, query_nodes=None, top_k=3, max_attempts=10):
        def is_obj_at_node(node):
            for child in node.get_children_nodes():
                if child.description == self.goal_node.description:
                    return True
            return False

        # Iteratively ask agent for predictions and give it observations
        
        num_attempts = 1
        pred_node = action
        all_furniture_nodes = self.scene.scene_graph.get_nodes(NodeType.FURNITURE)
        if not pred_node:
            pred_node = random.choice(all_furniture_nodes)
        visited_nodes = []
        visited_node_ids = []
        acc_cost = 0
        curr_node = self.get_random_furniture_node()
        while num_attempts <= max_attempts:
            agent.receive_observation(pred_node)
            acc_cost += get_cost_between_nodes(curr_node, pred_node, self.scene.scene_graph)

            visited_nodes.append(pred_node)
            visited_node_ids.append(pred_node.unique_id)

            if is_obj_at_node(pred_node):
                reward = num_attempts
                obs = pred_node
                info = {'acc_cost': acc_cost, 'visited_nodes': visited_nodes}
                done = True
                return obs, reward, done, info
            curr_node = pred_node # pred_node becomes the next "current node" for the agent.
            pred_node = agent.make_predictions(TaskType.FIND_OBJECT, current_scene, query_nodes, ignore_nodes_to_pred=visited_node_ids)
            if pred_node is None: # Could not find non-visited node, choose a random non-visited node
                filtered_options = [node for node in all_furniture_nodes if not node.unique_id in visited_node_ids]
                pred_node = random.choice(filtered_options)
            num_attempts += 1

        # object not found
        reward = num_attempts
        obs = pred_node
        info = {'acc_cost': acc_cost, 'visited_nodes': visited_nodes}
        done = True 
        return obs, reward, done, info

    def get_task_type(self):
        return TaskType.FIND_OBJECT

class PredictDynamicsTask(Task):
    def reset(self):
        moved_objects = super().reset()
        query = self.pick_query(moved_objects, 6)
        return query 

    def step(self, action, agent_type=None, top_k=3):
        # gather ground truth
        goal_object_nodes = list(action.keys())
        total_loss_edge_prob = 0.0
        max_likelihood_parents = []
        objs_with_no_edges = 0
        for object_node in goal_object_nodes:
            option_probs = self.scene_evolver.get_move_target_probs(object_node)
            if len(option_probs) == 0:
                objs_with_no_edges += 1
                continue
            gt_edges = {}
            # post process to a dict with top_k entities and only the node as the key
            edge_prob = dict(
                sorted(option_probs.items(), key=lambda item: item[1], reverse=True)[:top_k]
            )
            gt_edges = {k[0]:v for (k,v) in edge_prob.items()}
            max_prob_node = list(gt_edges.keys())[0]
            max_likelihood_parents.append(max_prob_node)
            # compute loss
            predicted_edges = action[object_node]['edge_prob']
            loss_edge_prob_i = 0.0
            if agent_type is None:
                raise ValueError("Predict Task Dynamics expects agent type in the scene step.")
            get_node_id = lambda node: node.description if agent_type == "priors" else node.unique_id
            gt_node_ids = [get_node_id(n) for n in gt_edges.keys()]
            pred_node_ids = [get_node_id(n) for n in predicted_edges.keys()]
            gt_length = len(gt_node_ids) # this is also the highest possible index distance (index == max_length if index is not in )
            for i in range(gt_length):
                node_id = gt_node_ids[i]
                gt_index = i
                if node_id in pred_node_ids:
                    pred_ind = pred_node_ids.index(node_id)
                    loss_edge_prob_i += (abs(gt_index - pred_ind) / gt_length)
                else:
                    loss_edge_prob_i += 1.0
            # OLD PROBABILISTIC METHOD
            # if agent_type == "priors":
            #     pred_node_by_identifier = {node.description:node for node in predicted_edges.keys()}
            #     gt_node_by_identifier = {node.label: node for node in gt_edges.keys()}
            # else:
            #     pred_node_by_identifier = {node.unique_id:node for node in predicted_edges.keys()}
            #     gt_node_by_identifier = {node.unique_id: node for node in gt_edges.keys()}
            # for furniture_node in gt_edges:
            #     if agent_type == "priors":
            #         node_key = furniture_node.label
            #     else:
            #         node_key = furniture_node.unique_id
            #     if node_key in pred_node_by_identifier:
            #         pred_node = pred_node_by_identifier[node_key]
            #         pred_prob, gt_prob = predicted_edges[pred_node], gt_edges[furniture_node]
            #         loss_edge_prob_i += abs(pred_prob - gt_prob)
            #     else:  # furniture node not predicted
            #         loss_edge_prob_i += 1.0
            # for (id, pred_node) in pred_node_by_identifier.items():
            #     if not id in gt_node_by_identifier: # predicted but not in GT
            #         loss_edge_prob_i += 1.0
            loss_edge_prob_i /= len(gt_edges)  # len of gt_edges <= top_k
            total_loss_edge_prob += loss_edge_prob_i
        total_loss_edge_prob /= (len(goal_object_nodes) - objs_with_no_edges)  # average over all goal obj nodes
        info = {}
        reward = 1 - total_loss_edge_prob
        done = True
        if len(max_likelihood_parents) != 0:
            obs = max_likelihood_parents
        else:
            obs = [self.get_random_furniture_node()]
        return obs, reward, done, info

    def get_task_type(self):
        return TaskType.PREDICT_ENV_DYNAMICS

def make_task(scene_sampler, scene_evolver, task_type, eps_per_scene, num_objs_to_predict=1):
    if task_type == TaskType.PREDICT_LOC:
        return PredictLocationsTask(scene_sampler, scene_evolver, eps_per_scene=eps_per_scene, num_objs_to_predict=num_objs_to_predict)
    elif task_type == TaskType.PREDICT_ENV_DYNAMICS:
        return PredictDynamicsTask(scene_sampler, scene_evolver, eps_per_scene=eps_per_scene)
    elif task_type == TaskType.FIND_OBJECT:
        return FindObjTask(scene_sampler, scene_evolver, eps_per_scene=eps_per_scene)
    else:
        raise ValueError("Task Type {} not implemented".format(task_type.value))
        
        
