import hydra
import memsearch
from memsearch.igridson_utils import *
from mini_behavior.window import Window
from memsearch.tasks import TaskType, make_task
from tqdm import tqdm 
import multiprocessing
import functools
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from enum import Enum

class VizMode(Enum):
    """
    Three visualization modes are supported:
    HEADLESS: igridson environment will not be rendered at all. 
        Use this when you don't want any visualizations and only wish to calculate path length
    RENDER: igridson environment will be rendered and visualized in a pop-up window in real time.
        Frames will be saved to the specified directory
    """
    HEADLESS = 0
    RENDER = 1

def eval_agent(cfg, viz_mode=VizMode.HEADLESS, agent_type=None):
    # scene sampler and evolver
    scene_sampler, scene_evolver = make_scene_sampler_and_evolver_(cfg.scene_gen)
    scene = scene_sampler.sample()
    agent = make_agent_type(cfg, agent_type, TaskType.FIND_OBJECT, scene_sampler, scene_evolver)
    # The task env will manage the agent, and make predictions in a symbolic space using the scene graph
    task = make_task(scene_sampler, scene_evolver, TaskType.FIND_OBJECT, cfg.task.eps_per_scene)
    # The scene graph is shared by the igridson environment too
    env = SMGFixedEnv(scene_sampler, scene_evolver, scene=scene, env_evolve_freq=-1)

    if viz_mode == VizMode.RENDER:
        window = Window(f'Memory Object Search -- agent type {agent_type}')
    else:
        window = None
    # Jointly reset task and igridson env
    reset(env, window, task, agent)
    
    print("Starting exp", agent_type)

    avg_reward = 0.0
    num_successes = 0
    num_steps = 0

    all_rewards, all_steps = [], []

    for _ in tqdm(range(cfg.num_queries)):
        # Generate a query using igridson env
        query = get_query(env)
        # Simulate the agent using the task env and get all the visited nodes
        pred_node, score, done, info = simulate_agent(agent, env, query, task, max_attempts=cfg.max_attempts)
        visited_nodes = info['visited_nodes']
        
        # Next, run the visited nodes through the igridson env to calvulate A* length
        # and visualize if specified
        if viz_mode == VizMode.HEADLESS:
            curr_reward = get_astar_path(env, visited_nodes)
        else:
            curr_reward = get_astar_path(
                env, 
                visited_nodes,
                window,
                save_dir='./outputs/igridson_simulations',
                exp_name=cfg.run_name
            )
        all_rewards.append(curr_reward)
        avg_reward += curr_reward

        num_steps += score
        all_steps.append(score)

        if score != (cfg.max_attempts + 1):
            num_successes += 1
        
        # Jointly reset igridson and task envs. 
        # This will force an evolution of the scene graph
        reset(env, window, task, agent)

    num_steps /= cfg.num_queries
    avg_reward /= cfg.num_queries
    success_rate = num_successes / cfg.num_queries
    steps_std_dev = np.std(all_steps)
    reward_std_dev = np.std(all_rewards)

    # Plot and save histograms
    plt.suptitle(agent_type)
    plt.subplot(121)
    plt.hist(all_rewards)
    plt.title("Path Length {:.2f} +/- {:.2f}".format(avg_reward, reward_std_dev))

    plt.subplot(122)

    plt.hist(all_steps, np.arange(1, cfg.max_attempts, 1))
    plt.title("Num Attempts {:.2f} +/- {:.2f}".format(num_steps, steps_std_dev))
    plt.tight_layout(pad=2.0)

    save_p = Path(f"./outputs/{cfg.run_name}/plots")
    save_p.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_p / f"{agent_type}.png"))

    return {agent_type: {"Cost": {"mean": avg_reward, "std": reward_std_dev, 'distribution': all_rewards}, "Num Steps": {"mean": num_steps, "std": steps_std_dev, 'distribution': all_steps}, "avg success rate": success_rate}}

@hydra.main(version_base=None, config_path=memsearch.CONFIG_PATH, config_name="config")
def main(cfg):
    agent_types = cfg.agents.agent_types
    if cfg.viz_mode == "headless":
        viz_mode = VizMode.HEADLESS
    elif cfg.viz_mode == "render":
        viz_mode = VizMode.RENDER
    
    if viz_mode != VizMode.RENDER:
        multiprocessing.set_start_method('spawn')
        run_eval_agent = functools.partial(eval_agent, cfg, viz_mode)
        tqdm.set_lock(multiprocessing.RLock())
        with multiprocessing.Pool(processes = len(agent_types)) as pool:
            all_score_vecs = pool.map(run_eval_agent, agent_types)
    else:
        all_score_vecs = [eval_agent(cfg, viz_mode, agent_type) for agent_type in agent_types]
    
    rearranged_dict = {list(agent_d.keys())[0]: agent_d[list(agent_d.keys())[0]] for agent_d in all_score_vecs}
    metrics = list(rearranged_dict[agent_types[0]].keys())
    
    for metric in metrics:
        print(metric)
        for agent_type in agent_types:
            metric_val = rearranged_dict[agent_type][metric]
            if metric == "avg success rate":
                print("Agent: {}, Success Rate: {}".format(agent_type, metric_val))
            else:
                mean, std = metric_val['mean'], metric_val['std']
                print("Agent: {}, {:.2f} +/- {:.2f}".format(agent_type, mean, std))

    with open('FINAL_RESULTS.pkl', 'wb') as f:
        pickle.dump(all_score_vecs, f)

if __name__ == '__main__':
    main()  # type: ignore