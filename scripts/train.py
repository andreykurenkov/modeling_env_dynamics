import os
import hydra
import random
import memsearch
from memsearch.util import configure_logging
from memsearch.training import train, test
from memsearch.dataset import make_featurizers

def run_training(cfg):
    logger = configure_logging(name='train')
    node_featurizer, edge_featurizer = make_featurizers(cfg.model, False, cfg.task.num_steps)
    random.seed('training seed')
    model = train(cfg = cfg,
                  num_epochs = cfg.num_train_epochs,
                  node_featurizer = node_featurizer,
                  edge_featurizer = edge_featurizer,
                  add_num_nodes = cfg.model.add_num_nodes,
                  add_num_edges = cfg.model.add_num_edges,
                  use_edge_weights = cfg.model.use_edge_weights,
                  num_labels_per_batch = cfg.train_labels_per_batch,
                  logger = logger)
    random.seed('testing seed')
    test(cfg,
         model = model,
         node_featurizer = node_featurizer,
         edge_featurizer = edge_featurizer,
         add_num_nodes = cfg.model.add_num_nodes,
         add_num_edges = cfg.model.add_num_edges,
         use_edge_weights = cfg.model.use_edge_weights,
         num_labels_per_batch = cfg.test_labels_per_batch,
         logger = logger)

@hydra.main(version_base=None,
            config_path=memsearch.CONFIG_PATH,
            config_name="config")
def main(cfg):
    run_training(cfg)

if __name__ == "__main__":
    main() # type: ignore
