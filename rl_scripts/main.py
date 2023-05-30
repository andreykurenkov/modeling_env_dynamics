import hydra

import memsearch
from memsearch.igridson_env import SMGFixedEnv
from memsearch.igridson_utils import make_scene_sampler_and_evolver_
import matplotlib.pyplot as plt

def visualize(env, obs):
    full_grid = env.render(mode="rgb_array", tile_size=32)
    
    # Visualize observed and full grid
    plt.subplot(1,2,1)
    plt.title("Observed view")
    plt.imshow(obs["image"])
    plt.subplot(1,2,2)
    plt.title("Full Grid")
    plt.imshow(full_grid)
    plt.tight_layout()
    plt.show()
    env.reset()

def run_training(cfg):
    scene_sampler, scene_evolver = make_scene_sampler_and_evolver_(cfg.scene_gen)
    env = SMGFixedEnv(scene_sampler=scene_sampler, scene_evolver=scene_evolver)
    for _ in range(100):
        env.reset()
        for idx in range(10000):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action) # obs is a dict with keys "image", "direction" and "mission"
            if done:
                print(f"Done at {idx}")
                break


@hydra.main(version_base=None,
            config_path=memsearch.CONFIG_PATH,
            config_name="config")
def main(cfg):
    run_training(cfg)

if __name__ == "__main__":
    main() # type: ignore

