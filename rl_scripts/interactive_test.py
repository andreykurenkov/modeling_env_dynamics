import hydra

import memsearch
from memsearch.igridson_env import SMGFixedEnv
from memsearch.igridson_utils import make_scene_sampler_and_evolver_
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys 

matplotlib.rcParams.update({'font.size': 4})

KEYBOARD_ACTION_MAP = {'w': 2, 'a': 0, 'd': 1}
        
def visualize(env, obs, reward, done):
    full_grid = env.render(mode="rgb_array", tile_size=32)
    # Visualize observed and full grid
    plt.subplot(1,2,1)
    plt.title("Observed view")
    plt.imshow(obs["image"])
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title("Full Grid")
    plt.imshow(full_grid)
    plt.tight_layout()
    plt.suptitle("Goal: {}, Reward: {}, Done: {}".format(env.goal_obj_label, reward, done))
    plt.axis('off')
    plt.show()

def run_training(cfg):
    scene_sampler, scene_evolver = make_scene_sampler_and_evolver_(cfg.scene_gen)
    env = SMGFixedEnv(scene_sampler=scene_sampler, scene_evolver=scene_evolver, set_goal_icon=True, env_evolve_freq=5)
    
    def on_press(event):
        print("Target poses:", env.target_poses)
        print("Agent pose:", env.agent_pos)
        sys.stdout.flush()
        if event.key == 'r':
            print("Resetting...")
            env.reset()
            print("done.")
            init_action = env.action_space.sample()
            obs, reward, done, info = env.step(init_action)
            visualize(env, obs, reward, done)
        elif event.key in ['w', 'a', 'd']:
            action = KEYBOARD_ACTION_MAP[event.key]
            obs, reward, done, info = env.step(action)
            visualize(env, obs, reward, done)
        else:
            print("Invalid input", event.key)
    fig = plt.gcf()
    fig.canvas.mpl_connect('key_press_event', on_press)
    
    #Initialize
    init_action = env.action_space.sample()
    obs, reward, done, info = env.step(init_action)
    visualize(env, obs, reward, done)

@hydra.main(version_base=None,
            config_path=memsearch.CONFIG_PATH,
            config_name="config")
def main(cfg):
    run_training(cfg)

if __name__ == "__main__":
    main() # type: ignore