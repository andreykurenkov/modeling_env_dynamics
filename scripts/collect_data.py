import hydra
import os
import gc
import memsearch
import logging

from hydra.core.hydra_config import HydraConfig
from memsearch.util import configure_logging
from memsearch.tasks import TaskType
from memsearch.running import collect_data
from memsearch.dataset import make_featurizers, SGMDataset

logger = logging.getLogger(__name__)

def convert_to_pyg(cfg, node_featurizer, edge_featurizer, sgm_graphs, data_type):
    input_data_path = f'{cfg.collect_data_dir}/{data_type}/'
    output_data_path = f'{cfg.processed_dataset_dir}/{data_type}/'
    for is_heterogenous in [False, True]:
        if is_heterogenous:
            print(f'Creating tensors for heterogenous {data_type} graphs')
        else:
            print(f'Creating tensors for homogenous {data_type} graphs')
        # filter out super big graphs to make things faster
        sgm_graphs = [graph for graph in sgm_graphs if len(graph.get_edges()) < 2500  
                                                    and len(graph.nodes) < 225]
        dataset = SGMDataset(input_data_path,
                             output_data_path,
                             node_featurizer,
                             edge_featurizer,
                             include_labels=True,
                             add_num_nodes = cfg.model.add_num_nodes,
                             add_num_edges = cfg.model.add_num_edges,
                             num_workers=cfg.process_data_num_workers,
                             heterogenous=is_heterogenous,
                             reverse_edges=cfg.model.reversed_edges,
                             data = sgm_graphs)
        del dataset #this does processing upon construction, no need to store

def run_data_collection(cfg):
    logger = configure_logging(name='collet_data')
    output_dir = HydraConfig.get().runtime.output_dir
    configure_logging(cfg.log_path)
    save_graphs_path=cfg.collect_data_dir
    if cfg.process_graphs_after_collection:
        node_featurizer, edge_featurizer = make_featurizers(cfg.model, True, cfg.task.num_steps)
    if not cfg.process_graphs_after_collection and not os.path.isdir(save_graphs_path):
        os.makedirs(save_graphs_path+'/train')
        os.makedirs(save_graphs_path+'/test')
    if not cfg.process_graphs_after_collection and len(os.listdir(save_graphs_path+'/train')) == cfg.data_gen.num_steps_train \
       or cfg.process_graphs_after_collection and os.path.isdir(cfg.processed_dataset_dir+'/train'):
        print('Data already collected, quitting.')
        return 
    cfg.agents.memorization_use_priors = True
    print('Collecting train data with agent type %s\n'%cfg.data_gen.agent_type)
    train_sgms = collect_data(cfg,
                 logger,
                 output_dir,
                 num_steps = cfg.data_gen.num_steps_train,
                 agent_type = cfg.data_gen.agent_type,
                 save_graphs_path = '%s/train'%(save_graphs_path),
                 save_sgm_graphs = not cfg.process_graphs_after_collection)
    if cfg.process_graphs_after_collection:
        convert_to_pyg(cfg, node_featurizer, edge_featurizer, train_sgms, 'train')
    del train_sgms
    gc.collect()
    print('\nDone!')

    print('\nCollecting test data with agent type %s\n'%cfg.data_gen.agent_type)
    test_sgms = collect_data(cfg,
                 logger,
                 output_dir,
                 num_steps = cfg.data_gen.num_steps_test,
                 agent_type=cfg.data_gen.agent_type,
                 save_graphs_path='%s/test'%(save_graphs_path),
                 save_sgm_graphs = not cfg.process_graphs_after_collection)
    print('\nDone!')

    if cfg.process_graphs_after_collection:
        convert_to_pyg(cfg, node_featurizer, edge_featurizer, test_sgms, 'test')
    del test_sgms

@hydra.main(version_base=None, config_path=memsearch.CONFIG_PATH, config_name="config")
def main(cfg):
    run_data_collection(cfg)

if __name__ == '__main__':
    main() #type: ignore
