default_configs = {
    'defaults': [{'/scene_gen': 'large_scenes'}, {'/agents': 'all'}, {'/data_gen': '10k'}, {'/task': 'predict_location'}, {'/model': 'gcn'}], 
    'run_name': 'pl_l_d_d_l_za_n', 
    'dataset': 'pl_l_d_d_l_za_n', 
    'node_features': ['word_vec', 'time_since_observed', 'times_observed', 'time_since_state_change', 'node_type'], 
    'edge_features': ['time_since_observed', 'time_since_state_change', 'times_observed', 'times_state_true', 'last_observed_state', 'freq_true', 'edge_type', 'prior_prob'], 
    'agents': 
        {
            'agent_priors_type': 'detailed', 
            'psg_use_priors': True
        }, 
    'scene_gen': 
        {
            'scene_priors_type': 'detailed', 
            'priors_noise': 0.2,
            'priors_sparsity_level': 0.4
        }
    }

TASK_KEY_MATCHES = {
    'pl': 'predict_location',
    'pd': 'predict_env_dynamics',
    'ps': 'predict_scene'
}
SCENE_SIZE_KEY_MATCHES = {
    'l': 'large_scenes',
    's': 'small_scenes'
}
SCENE_PRIORS_KEY_MATCHES = {
    'd': 'detailed',
    'c': 'coarse',
    'r': 'random'
}
AGENT_PRIORS_KEY_MATCHES = {
    'd': 'detailed',
    'c': 'coarse',
    'r': 'random'
}
DATASET_SIZE_KEY_MATCHES = {
    'l': '10k',
    's': '5k',
    't': '1k'
}
PRIORS_NOISE_KEY_MATCHES = {
    'za': 0.2,
    'n': 0
}
PRIORS_SPARSITY_LEVEL_KEY_MATHES = {
    'za': 0.2,
    'n': 0.2
}

DEFAULT_EXP_NAMES = ['pl_l_d_d_l_za_n',
    'ps_l_d_d_l_za_n',
    'pd_l_d_d_l_za_n',
    'pl_l_d_c_l_za_n',
    'ps_l_d_c_l_za_n',
    'pd_l_d_c_l_za_n',
    'pl_l_d_r_l_za_n',
    'ps_l_d_r_l_za_n',
    'pd_l_d_r_l_za_n',
    'pl_s_d_d_l_za_n',
    'ps_s_d_d_l_za_n',
    'pd_s_d_d_l_za_n',
    'pl_s_d_c_l_za_n',
    'ps_s_d_c_l_za_n',
    'pd_s_d_c_l_za_n',
    'pl_s_d_r_l_za_npp',
    'ps_s_d_r_l_za_npp',
    'pd_s_d_r_l_za_npp',
    'pl_l_d_d_l_za_npp',
    'ps_l_d_d_l_za_npp',
    'pd_l_d_d_l_za_npp',
    'pl_l_d_c_l_za_npp',
    'ps_l_d_c_l_za_npp',
    'pd_l_d_c_l_za_npp',
    'pl_l_d_r_l_za_npp',
    'ps_l_d_r_l_za_npp',
    'pd_l_d_r_l_za_npp',
    'pl_s_d_r_l_za_npp',
    'ps_s_d_r_l_za_npp',
    'pd_s_d_r_l_za_npp',
    'pl_l_d_d_l_za_nwv',
    'ps_l_d_d_l_za_nwv',
    'pd_l_d_d_l_za_nwv',
    'pl_l_d_c_l_za_nwv',
    'ps_l_d_c_l_za_nwv',
    'pd_l_d_c_l_za_nwv',
    'pl_l_d_r_l_za_nwv',
    'ps_l_d_r_l_za_nwv',
    'pd_l_d_r_l_za_nwv',
    'pl_l_d_d_l_za_ntf',
    'ps_l_d_d_l_za_ntf',
    'pd_l_d_d_l_za_ntf',
    'pl_l_d_c_l_za_ntf',
    'ps_l_d_c_l_za_ntf',
    'pd_l_d_c_l_za_ntf',
    'pl_l_d_r_l_za_ntf',
    'ps_l_d_r_l_za_ntf',
    'pd_l_d_r_l_za_ntf',
    'pl_l_d_d_l_n_n',
    'ps_l_d_d_l_n_n',
    'pd_l_d_d_l_n_n',
    'pl_l_d_c_l_n_n',
    'ps_l_d_c_l_n_n',
    'pd_l_d_c_l_n_n',
    'pl_l_d_r_l_n_n',
    'ps_l_d_r_l_n_n',
    'pd_l_d_r_l_n_n',
    'pl_s_d_d_l_n_n',
    'ps_s_d_d_l_n_n',
    'pd_s_d_d_l_n_n',
    'pl_s_d_c_l_n_n',
    'ps_s_d_c_l_n_n',
    'pd_s_d_c_l_n_n',
]
