import hydra

import memsearch
from memsearch.igridson_env import SMGFixedEnv
from memsearch.igridson_utils import make_scene_sampler_and_evolver_, reset, visualize_agent, evolve

import matplotlib.pyplot as plt
from ray.rllib.utils import check_env

@hydra.main(version_base=None,
            config_path=memsearch.CONFIG_PATH,
            config_name="config")
def main(cfg):
    scene_sampler, scene_evolver = make_scene_sampler_and_evolver_(cfg.scene_gen)
    env = SMGFixedEnv(scene_sampler=scene_sampler, scene_evolver=scene_evolver)
    breakpoint()

    check_env(env)

if __name__ == "__main__":
    main() # type: ignore

