import os
from memsearch.scene import *
from mini_behavior.utils.navigate import *
from memsearch.igridson_env import *
from mini_behavior.minibehavior import MiniBehaviorEnv
from memsearch.tasks import TaskType
from memsearch.agents import make_agent
from memsearch.dataset import make_featurizers

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


def make_agent_type(cfg, agent_type, task_type, scene_sampler, scene_evolver):
    node_featurizer, edge_featurizer = make_featurizers(cfg.model, False, cfg.task.num_steps)
    if 'sgm' in agent_type:
        model_config = cfg.model
    else:
        model_config = None
    agent = make_agent(cfg.agents, agent_type, task_type, node_featurizer, edge_featurizer, scene_sampler,
                       scene_evolver, False, model_config)
    return agent


def get_query(env):
    all_object_nodes = env.scene.scene_graph.get_nodes_with_type(NodeType.OBJECT)
    sample_from = all_object_nodes if len(env.moved_objs) == 0 else env.moved_objs
    return random.choice(sample_from)


def simulate_agent(agent, env, query, task, max_attempts=5):
    current_scene = env.scene_evolver.scene
    task.goal_node = query
    env.set_mission_by_node(query)

    prediction = agent.make_predictions(TaskType.FIND_OBJECT, current_scene, query)
    obs, score, done, info = task.step(prediction, agent, current_scene, query, max_attempts=max_attempts)
    agent.receive_observation(obs)
    agent.step()

    return obs, score, done, info


def evolve(env, window, agent, task):
    env.evolve()
    redraw(env, window)

    current_scene = env.scene_evolver.scene
    agent.transition_to_new_scene(current_scene)
    task.scene = env.scene_evolver.scene
    for edge in task.scene.scene_graph.get_edges():
        edge.age += 1
    for node in env.moved_objs:
        node.times_moved += 1
    task.current_ep += 1


def step(action, env, window):
    obs, reward, done, info = env.step(action)
    redraw(env, window)


def is_obj_at_node(node, goal_node):
    for child in node.get_children_nodes():
        if child.description == goal_node.description:
            return True
        return False

def get_astar_path(env, visited_nodes, window=None, save_dir=None, exp_name=None):
    def choose_node(node):
        if type(node) == PriorsNode or type(node) == SGMNode:
            raise ValueError("node of type {} found. Expected scene graph node".format(type(node)))
        else:
            return node
    goal_instance = env.node_to_obj[env.goal_node]
    goal_instance.icon_color = 'green'
    env.place_agent(i=0, j=0)
    env.step_count = 0
    curr_agent_pos = None
    total_path_length = 0

    for node in visited_nodes:
        obj_node = choose_node(node)
        path, actions, end_pos = get_path_and_actions(env, obj_node, curr_agent_pos)
        curr_agent_pos = end_pos
        total_path_length += len(path)
         
        if window:
            save_path = get_save_path(save_dir, [exp_name, obj_node.unique_id])
            for action in actions:
                step(MiniBehaviorEnv.Actions(action), env, window)
                save_img(window, os.path.join(save_path, f'x_pos{end_pos[0]}_y_pos{end_pos[1]}_step_{env.step_count}.png'))

    return total_path_length

def get_path_and_actions(env, target_node, agent_pos=None):
    maze = env.grid.get_maze()
    obj_instance = env.node_to_obj[target_node]

    if not agent_pos:
        start_pos = env.agent_pos
    else:
        start_pos = agent_pos
    if type(obj_instance) == FurnitureObj:
        i, j = random.choice(obj_instance.all_pos)
    else:
        i, j = obj_instance.cur_pos
    end_pos = get_pos_next_to_obj(env, i, j)
    assert end_pos is not None, 'not able to reach obj {}, {}'.format(i, j)

    start_room = env.room_from_pos(*(start_pos[1], start_pos[0]))
    end_room = env.room_from_pos(*(end_pos[1], end_pos[0]))
    path = navigate_between_rooms(start_pos, end_pos, start_room, end_room, maze)
    actions = get_actions(env.agent_dir, path)
    return path, actions, end_pos

def nodes_to_coords(env, nodes):
    coods = []
    for node in nodes:
        obj_instance = env.node_to_obj[node]
        if type(obj_instance) == FurnitureObj:
            i, j = random.choice(obj_instance.all_pos)
        else:
            i, j = obj_instance.cur_pos
        coods.append((i,j))
    return coods 

def cumulative_manhattan_distance(visited_nodes):
    total_dist = 0.0
    manhattan_dist = lambda x : (abs(x[0][0] - x[1][0]) + abs(x[0][1] - abs(x[1][1])))
    for i in range(len(visited_nodes) - 1):
        total_dist += manhattan_dist([visited_nodes[i], visited_nodes[i+1]])
    return total_dist

def get_pos_next_to_obj(env, i, j):
    maze = env.grid.get_maze()
    for pos in [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j)]:
        # if env.grid.is_empty(*pos):
        #     return pos
        if maze[pos[1]][pos[0]] == 0:
            return tuple(pos)
    return None


def get_save_path(save_dir, save_name):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    path = save_dir
    for subdir in save_name:
        path = os.path.join(path, subdir)
        if not os.path.exists(path):
            os.mkdir(path)
    return path


def make_scene_sampler_and_evolver_(cfg):
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
                                       sparsity_level=cfg.priors_sparsity_level,
                                       room_layout=DEFAULT_ROOM_LAYOUT)
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



TILE_PIXELS = 32


def redraw(env, window):
    if window is not None:
        img = env.render('rgb_array', tile_size=TILE_PIXELS)
        window.no_closeup()
        if hasattr(env, 'goal_node'):
            caption = f"Goal: {env.goal_obj_label}"
            window.set_caption(caption)
        window.show_img(img)
        
def save_img(window, pathname='saved_grid.png'):
    window.save_img(pathname)


def reset(env, window, task, agent):
    obs = env.reset()
    agent.transition_to_new_scene(env.scene_evolver.scene)
    task.scene = env.scene_evolver.scene
    for edge in task.scene.scene_graph.get_edges():
        edge.age += 1
    for node in env.moved_objs:
        node.times_moved += 1
    task.current_ep += 1

    if window is not None:
        redraw(env, window)

    return obs


def check_graph_and_grid(env):
    graph = env.scene.scene_graph
    node_to_obj = env.node_to_obj
    for furniture_node in graph.get_nodes_with_type(NodeType.FURNITURE):
        for obj_node in [node for node in furniture_node.get_children_nodes() if node.type == NodeType.OBJECT]:
            obj = node_to_obj[obj_node]
            furniture = node_to_obj[furniture_node]

            edge = [e for e in obj_node.edges if e.node1 == furniture_node or e.node2 == furniture_node][0]
            edge_type = edge.type if edge.node1 == obj_node else RECIPROCAL_EDGE_TYPES[edge.type]
            check_state(env, obj, furniture, edge_type)


def check_state(env, obj, furniture, edge_type):
    if edge_type.value == "in":
        assert obj.check_rel_state(env, furniture, 'inside')
    elif edge_type.value == "contains":
        assert furniture.check_rel_state(env, obj, 'inside')
    else:
        assert obj.check_rel_state(env, furniture, edge_type.value)
