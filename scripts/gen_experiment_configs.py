from memsearch.experiment_configs import *
import itertools
from tqdm import tqdm
import copy
import yaml
import argparse

default_configs = {
    'defaults': [{'/scene_gen': 'large_scenes'}, {'/agents': 'all'}, {'/data_gen': '10k'}, {'/task': 'predict_location'}, {'/model': 'heat'}], 
    'run_name': 'pl_l_dn_d_d_za_n', 
    'model': 
        {
            'include_transformer': True,
            'node_features': 'all',
            'edge_features': 'all',
        }, 
    'agents': 
        {
            'agent_priors_type': 'detailed', 
            'sgm_use_priors': True
        }, 
    'scene_gen': 
        {
            'add_or_remove_objs': True,
            'scene_priors_type': 'detailed', 
            'scene_priors_type': 'detailed', 
            'priors_noise': 0.25,
            'priors_sparsity_level': 0.25
        }
    }

TASK_MATCHES = {
    'pl': 'predict_location',
    'pls': 'predict_locations',
    'pd': 'predict_env_dynamics',
    'fo': 'find_object'
}
SCENE_SIZE_MATCHES = {
    'l': 'large_scenes',
    's': 'small_scenes'
}
SCENE_NODE_DYNAMICS_MATCHES = {
    'dn': True, #dynamic nodes
    'sn': False #static nodes
}
SCENE_PRIORS_MATCHES = {
    'd': 'detailed',
    'c': 'coarse',
}
AGENT_PRIORS_MATCHES = {
    'd': 'detailed',
    'c': 'coarse',
}
PRIORS_NOISE_MATCHES = {
    'za': 0.25,
    'n': 0
}
PRIORS_SPARSITY_LEVEL_MATCHES = {
    'za': 0.25,
    'n': 0.25
}
MODEL_INCLUDE_TRANSFORMER_MATCHES = {
    'it': True,
    'et': False
}

ALL_EXP_NAMES = ['_'.join(l) for l in itertools.product(TASK_MATCHES.keys(), 
        #SCENE_SIZE_MATCHES.keys(), 
        SCENE_NODE_DYNAMICS_MATCHES.keys(), 
        SCENE_PRIORS_MATCHES.keys(),
        AGENT_PRIORS_MATCHES.keys(),
        PRIORS_NOISE_MATCHES.keys(),
        MODEL_INCLUDE_TRANSFORMER_MATCHES.keys(),
        ['n','nwv','ntf', 'npp'])]

def assign_new_value(cfg_dict, primary_key, cfg_key, value_match, secondary_key=None):
    if cfg_key not in value_match:
        raise ValueError("Incorrect config {}. Looking for one of {}".format(cfg_key, list(value_match.keys())))
    if secondary_key is None:
        cfg_dict[primary_key] = value_match[cfg_key]
    else:
        if isinstance(cfg_dict[primary_key], list):
            dict_list = cfg_dict[primary_key]
            match_found = False
            for i, sub_dict in enumerate(dict_list):
                if secondary_key in sub_dict:
                    cfg_dict[primary_key][i][secondary_key] = value_match[cfg_key]
                    match_found = True 
            if not match_found:
                raise ValueError("Couldn't find {} in {}".format(secondary_key, dict_list))
        else:
            cfg_dict[primary_key][secondary_key] = value_match[cfg_key]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_names_file",
        help="Path to text file containing the strings to be used to generate config files.",
        required=False,
        default=None
    )
    parser.add_argument(
        "--print_formatted_strings",
        help="Prints the formatted strings of the generated config files.",
        action="store_true"
    )
    parser.add_argument(
        "--exp_names",
        help="Comma separated experiment names. Defaults generated if none given",
        default=None
    )
    args = parser.parse_args()
    if args.exp_names_file:
        with open(args.exp_names_file, "r") as f:
            filenames = f.readlines()
    elif args.exp_names:
        filenames = args.exp_names.split(',')
    else:
        filenames = ALL_EXP_NAMES
    for filename in tqdm(filenames):
        filename = filename.strip()
        if args.print_formatted_strings:
            print(filename)
        config_keys = filename.split("_")
        config_dict = copy.deepcopy(default_configs)
        task_type, node_dynamics, scene_priors, agent_priors, priors_noise, include_transformer, ablations = config_keys
        assign_new_value(config_dict, "defaults", task_type, TASK_MATCHES, "/task")
        #assign_new_value(config_dict, "defaults", scene_size, SCENE_SIZE_MATCHES, "/scene_gen")
        assign_new_value(config_dict, "scene_gen", node_dynamics, SCENE_NODE_DYNAMICS_MATCHES, "add_or_remove_objs")
        assign_new_value(config_dict, "scene_gen", scene_priors, SCENE_PRIORS_MATCHES, "scene_priors_type")
        assign_new_value(config_dict, "agents", agent_priors, AGENT_PRIORS_MATCHES, "agent_priors_type")
        assign_new_value(config_dict, "scene_gen", priors_noise, PRIORS_NOISE_MATCHES, "priors_noise")
        assign_new_value(config_dict, "scene_gen", priors_noise, PRIORS_SPARSITY_LEVEL_MATCHES, "priors_sparsity_level")
        assign_new_value(config_dict, "model", include_transformer, MODEL_INCLUDE_TRANSFORMER_MATCHES, "include_transformer")
        
        # Ablations
        if ablations == "npp":
            config_dict['agents']['sgm_use_priors'] = False 
        elif ablations == "nwv":
            config_dict["model"]["node_features"] = ['time_since_observed', 'times_observed', 'time_since_state_change', 'state_change_freq', 'node_type']
            config_dict["model"]["edge_features"] = ['time_since_observed', 'time_since_state_change', 'times_observed', 'times_state_true', 'last_observed_state', 'freq_true', 'edge_type']
        elif ablations == "ntf":
            config_dict["model"]["node_features"] = ['text_embedding', 'node_type']
            config_dict["model"]["edge_features"] = ['cosine_similarity', 'last_observed_state', 'freq_true', 'prior_prob', 'edge_type']
        
        config_dict["run_name"] = filename
        config_dict["dataset"] = filename
        with open("configs/experiment/{}.yaml".format(filename), 'w') as file:
            file.write("# @package _global_\n")
        with open("configs/experiment/{}.yaml".format(filename), 'a') as file:
            documents = yaml.dump(config_dict, file, sort_keys=False)
