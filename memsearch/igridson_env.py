import numpy as np
from gym import spaces
import gensim.downloader as api

from mini_behavior.sampling import *
from mini_behavior.envs.fixed_scene import FixedEnv
from mini_behavior.objects import OBJECT_TO_IDX, IDX_TO_OBJECT
from mini_behavior.grid import TILE_PIXELS

from memsearch.graphs import NodeType, RECIPROCAL_EDGE_TYPES

TILE_PIXELS = 32

EDGE_TYPE_TO_FUNC = {
    "onTop": put_ontop,
    "in": put_inside,
    "contains": put_contains,
    "under": put_under,
}


class SMGFixedEnv(FixedEnv):
    def __init__(self, scene_sampler, scene_evolver, encode_obs_im=False, mission_mode='one_hot', scene=None, set_goal_icon=False, env_evolve_freq=100):
        self.scene_sampler = scene_sampler
        self.scene_evolver = scene_evolver

        self.scene = self.scene_sampler.sample() if scene is None else scene
        self.scene_evolver.set_new_scene(self.scene)

        self.env_evolove_freq = env_evolve_freq
        self.node_to_obj = None
        self.moved_objs = None

        num_objs = self.get_num_objs()
        self.mission_mode = mission_mode
        self.encode_obs_im = encode_obs_im

        self.initialized = False
        self.set_goal_icon = set_goal_icon

        super().__init__(num_objs=num_objs, agent_view_size=7)

    def validate_scene(self):
        # Check scene
        all_furniture_nodes = self.scene.scene_graph.get_nodes_with_type(NodeType.FURNITURE)
        
        for fn in all_furniture_nodes:
            obj_children = [node for node in fn.get_children_nodes() if node.type == NodeType.OBJECT]
            if len(obj_children) > 4:
                return False
        return True
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        # if env_evolve_freq = -1, don't evolve env during steps, only during resets
        if self.step_count > 1 and self.env_evolove_freq != -1 and self.step_count % self.env_evolove_freq == 0:
            self.evolve()
        return obs, reward, done, info
            
    def _reward(self):
        return -1
    
    def _gen_objs(self):
        super()._gen_objs()
        if self.node_to_obj is not None:
            self.graph_to_grid()

    def _set_obs_space(self):
        assert self.mission_mode in ['one_hot', 'word_vec', 'int'], "Only three modes supported: one hot, word vec or integer."
        
        if self.mission_mode == 'word_vec':
            self.word2vec_model = api.load("glove-twitter-25")
            mission_observation_space = spaces.Box(
                low=-1,
                high=1,
                shape=(25),
                dtype='float32'
            )
        elif self.mission_mode == 'int':
            mission_observation_space = spaces.Discrete(len(IDX_TO_OBJECT))
        elif self.mission_mode == 'one_hot':
            mission_observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(len(IDX_TO_OBJECT),),
                dtype='int'
            )
        else:
            assert "need valid obs mode for mission"
        if self.encode_obs_im: 
            image_observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.agent_view_size, self.agent_view_size, 3),
                dtype=np.uint8
            )
        else:
            image_observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.agent_view_size * TILE_PIXELS, self.agent_view_size * TILE_PIXELS, 3),
                dtype=np.uint8
            )

        self.observation_space = spaces.Dict({
            "direction": spaces.Box(low=0, high=4, shape=(), dtype=np.uint8),
            'image': image_observation_space,
            "mission": mission_observation_space,
        })
        

    def reset(self):
        # Hack around nightmare inheritance chain
        if not self.initialized:
            self._set_obs_space()
        
        # Reinitialize episode-specific variables
        self.agent_pos = (-1, -1)
        self.agent_dir = -1

        self.carrying = set()

        for obj in self.obj_instances.values():
            obj.reset()

        self.reward = 0

        # Generate a new random grid at the start of each episode
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # generate furniture view
        # self.furniture_view = self.grid.render_furniture(tile_size=TILE_PIXELS, obj_instances=self.obj_instances)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        assert self.grid.is_empty(*self.agent_pos)

        # Step count since episode start
        self.step_count = 0
        self.episode += 1

        # Make node to obj list
        self.set_node_to_obj()
        # TODO not sure
        if not self.initialized:
            self.graph_to_grid()
            self.initialized = True
        #TODO: Set the mission THIS IS RANDOM AND NEEDS TO GET FIXED
        self.set_random_mission()

        # Return first observation
        obs = self.gen_obs()

        self.reward = 0
        self.step_count = 0
        self.episode += 1

        if self.node_to_obj is not None:
            self.evolve()

        return obs

    def get_num_objs(self):
        num_objs = {}

        for node in self.scene.scene_graph.get_nodes_with_type(NodeType.FURNITURE) + self.scene.scene_graph.get_nodes_with_type(NodeType.OBJECT):
            num_objs[node.label] = num_objs.get(node.label, 0) + 1

        return num_objs

    def set_node_to_obj(self):
        """
        returns dict: key = node, value = obj_instance
        """
        self.node_to_obj = {}

        for obj_type, objs in self.objs.items():
            nodes = self.scene.scene_graph.get_nodes_with_label(obj_type)

            assert len(objs) == len(nodes)
            for i in range(len(objs)):
                self.node_to_obj[nodes[i]] = objs[i]

    def graph_to_grid(self):
        """
        NOTE: each edge obj has 1 parent node, and there are two edges between
        """
        # for every furniture
        for furniture_node in self.scene.scene_graph.get_nodes_with_type(NodeType.FURNITURE):
            furniture = self.node_to_obj[furniture_node]
            # for every obj related to the furniture
            for obj_node in furniture_node.get_children_nodes():
                if obj_node.type == NodeType.OBJECT:
                    obj = self.node_to_obj[obj_node]
                    edges = obj_node.get_edges_to_me()
                    if len(edges) > 0:
                        edge = edges[0]  # edge from obj to furniture
                        assert edge.node2 == obj_node and edge.node1 == furniture_node
                        EDGE_TYPE_TO_FUNC[RECIPROCAL_EDGE_TYPES[edge.type].value](self, obj, furniture)  # put the obj on the grid
                    else:
                        print("Found 0 length edges")

    def sample_to_grid(self, obj_node):
        if obj_node.type == NodeType.OBJECT:
            obj = self.node_to_obj[obj_node]
            if obj.cur_pos is not None and not obj.check_abs_state(state='inhandofrobot'):
                self.grid.remove(*obj.cur_pos, obj)

            edge = \
            [e for e in obj_node.edges if (e.node1.type == NodeType.FURNITURE or e.node2.type == NodeType.FURNITURE)][0]

            if edge.node1.type == NodeType.FURNITURE:
                furniture_node = edge.node1
                edge_type = RECIPROCAL_EDGE_TYPES[edge.type]
            else:
                furniture_node = edge.node2
                edge_type = edge.type

            furniture = self.node_to_obj[furniture_node]
            EDGE_TYPE_TO_FUNC[edge_type.value](self, obj, furniture)  # put the obj on the grid

            # uncomment for debugging
            # check_state(self, obj, furniture, edge_type)

    def evolve(self):
        # self.scene_evolver.scene = self.scene
        self.moved_objs = self.scene_evolver.evolve()  # list of objects that were moved
        for obj_node in self.moved_objs:
            if obj_node not in list(self.node_to_obj.keys()):  # if it is an added obj
                obj_instance = self.add_objs({obj_node.label: 1})[0]
                node_to_obj = self.node_to_obj
                node_to_obj[obj_node] = obj_instance
                self.node_to_obj = node_to_obj
                assert obj_node in list(self.node_to_obj.keys())
            self.sample_to_grid(obj_node)

    def _end_conditions(self):
        assert self.target_poses, "This function should only be called after set_mission"
        for target_pos in self.target_poses:
            if np.all(target_pos == self.front_pos) or self.step_count == self.max_steps:
                return True
        return False

    def set_mission(self, goal):
        """
        Sets the mission of the env
        """
        assert isinstance(goal, int) or isinstance(goal, str), "Expecting either obj index or obj name"
        
        if isinstance(goal, int): # Setting target by obj idx
            obj_label = IDX_TO_OBJECT[goal]
            obj_idx = goal
        elif isinstance(goal, str):
            obj_label = goal.lower()
            obj_idx = OBJECT_TO_IDX[obj_label]
        self.goal_obj_label = obj_label
        
        assert obj_label in self.objs.keys(), "Goal object not sampled in current scene."
        self.target_poses = [target_obj.cur_pos for target_obj in self.objs[obj_label]]
        
        # Set mission
        if self.mission_mode == 'one_hot':
            self.mission = np.eye(len(IDX_TO_OBJECT))[obj_idx]
        elif self.mission_mode == 'int':
            self.mission = obj_idx
        elif self.mission_mode == 'word_vec':
            model_inps = obj_label.split('_')
            vec = np.zeros((25))
            for inp in model_inps:
                vec += self.word2vec_model.get_vector(inp, norm=True)
            vec /= len(model_inps)
            self.mission = vec
        else:
            assert "Missing obs mode"
            
        if self.set_goal_icon: # Set icon of goal object to be green
            goal_objs = self.objs[obj_label]
            for goal_obj in goal_objs:
                goal_obj.icon_color = 'green'
            
    def get_possible_missions(self):
        all_object_nodes = self.scene.scene_graph.get_nodes_with_type(NodeType.OBJECT)
        all_obj_labels = [self.node_to_obj[node].type for node in all_object_nodes]
        return all_obj_labels
    
    def set_random_mission(self):
        all_goals = self.get_possible_missions()
        random_goal = random.choice(all_goals)
        self.set_mission(random_goal)
        
    def set_mission_by_node(self, node):
        self.goal_node = node 
        self.goal_obj_label = self.node_to_obj[node].type
        self.set_mission(self.goal_obj_label)
