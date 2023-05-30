import os
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.transforms.to_undirected import ToUndirected
#from torch_geometric.transforms import RandomLinkSplit
from hydra.core.hydra_config import HydraConfig
from memsearch.dataset import SGMDataset
from memsearch.models import make_model, create_path_to_model, compute_output
from tensorboardX import SummaryWriter
from collections import defaultdict

def sample_loss_edges_homogenous(data, max_to_sample, subsample=False, batch_by_node=True, reversed_edges=False):
    edge_labels = data.y.detach().numpy()
    edge_should_sample_for_loss = data.should_sample_for_loss.detach().numpy()
    if subsample:
        nonzero_indeces = edge_labels.nonzero()[0]
        nonzero_indeces = nonzero_indeces[edge_should_sample_for_loss[nonzero_indeces] == 1]
        flipped_edge_labels = 1 - edge_labels
        zero_indeces = flipped_edge_labels.nonzero()[0]
        zero_indeces = zero_indeces[edge_should_sample_for_loss[zero_indeces] == 1.0]
        num_to_sample = min([max_to_sample, len(zero_indeces), len(nonzero_indeces)])
        chosen_nonzero_indeces = random.sample(list(nonzero_indeces), int(num_to_sample/2))
        chosen_zero_indeces = random.sample(list(zero_indeces), int(num_to_sample/2))
        indeces = chosen_zero_indeces + chosen_nonzero_indeces
    else:
        indeces = edge_should_sample_for_loss.nonzero()[0]
    edges = data.edge_index[:,indeces]
    edge_features = data.edge_attr[indeces,:]
    labels = data.y[indeces]

    if labels.shape[0] <= 1:
        return [], [], torch.tensor([])

    if batch_by_node:
        batched_edges_dict = defaultdict(lambda: [])
        batched_edge_features_dict = defaultdict(lambda: [])
        batched_labels_dict = defaultdict(lambda: [])
        object_node_nums = set()
        for i in range(edges.size(dim=1)):
            if reversed_edges:
                object_node_num = edges[0,i].cpu().item()
            else:
                object_node_num = edges[1,i].cpu().item()
            object_node_nums.add(object_node_num)
            batched_edges_dict[object_node_num].append(edges[:,i])
            batched_edge_features_dict[object_node_num].append(edge_features[i])
            batched_labels_dict[object_node_num].append(labels[i].unsqueeze(0))
        batched_edges_tensors = [torch.stack(batched_edges_dict[node_num]) for node_num in object_node_nums]
        batched_edge_features_tensors = [torch.stack(batched_edge_features_dict[node_num]) for node_num in object_node_nums]
        batched_labels_tensors = [torch.stack(batched_labels_dict[node_num]) for node_num in object_node_nums]
        edges = pad_sequence(batched_edges_tensors, batch_first=True)
        edge_features = pad_sequence(batched_edge_features_tensors, batch_first=True)
        labels = torch.squeeze(pad_sequence(batched_labels_tensors, padding_value=2, batch_first=True))
    edges = edges.cuda()
    edge_features = edge_features.cuda()
    labels = labels.cuda()
    if len(labels.size()) == 1:
        labels = torch.unsqueeze(labels, dim=0)
    return edges, edge_features, labels

def sample_loss_edges_heterogenous(data, max_to_sample, subsample=False, batch_by_node=True, reversed_edges=True):
    edges_dict = {}
    edge_features_dict = {}
    labels_dict = {}
    for key in data.edge_index_dict:
        edges, edge_features, labels = sample_loss_edges_homogenous(data[key], max_to_sample, subsample, batch_by_node, reversed_edges)
        if labels.shape[0] <= 1:
            continue
        edges_dict[key] = edges
        edge_features_dict[key] = edge_features
        labels_dict[key] = labels
    return edges_dict, edge_features_dict, labels_dict

def sample_loss_edges(model, data, max_to_sample, subsample=False, batch_by_node=True, reversed_edges=True):
    if  model.is_heterogenous():
        return sample_loss_edges_heterogenous(data, max_to_sample, subsample, batch_by_node, reversed_edges)
    else:
        return sample_loss_edges_homogenous(data, max_to_sample, subsample, batch_by_node, reversed_edges)

def train_step_homogeneous(model,
        input_data,
        loss_compute_edges,
        loss_compute_edge_features,
        loss_compute_labels,
        criterion,
        optimizer,
        do_optim_step=True,
        edge_key=None):
    optimizer.zero_grad()
    out = compute_output(model, input_data, loss_compute_edges, loss_compute_edge_features, edge_key)
    if len(loss_compute_labels.size()) > 1:
        # if batched by node
        loss_mask = (loss_compute_labels != 2).long()
        loss_compute_labels = loss_compute_labels*loss_mask
        out = out*loss_mask
        num_zeros = torch.sum(torch.abs((loss_compute_labels-1))).item()
        num_ones = torch.sum(torch.abs((loss_compute_labels))).item()
        loss = criterion(out, loss_compute_labels)
        loss_compute_labels = loss_compute_labels.long()
        if num_ones != 0:
            # reduce loss associated with label 0 since there are more of those labels
            zero_loss_scaling = float(num_ones)/num_zeros
            loss[(1-loss_compute_labels).bool()] = loss[(1-loss_compute_labels).bool()]*zero_loss_scaling
        loss = torch.mean(loss)
    else:
        loss = criterion(out, loss_compute_labels)
    loss.backward()
    if do_optim_step:
        optimizer.step()
    accuracy = (out.argmax(dim=1) == loss_compute_labels.argmax(dim=1)).float().mean()
    return out, loss, accuracy

def train_step_heterogeneous(model,
        input_data,
        loss_compute_edges,
        loss_compute_edge_features,
        loss_compute_labels,
        criterion,
        optimizer):
    optimizer.zero_grad()
    outs = {}
    accuracies = []
    losses = []
    for key in loss_compute_labels:
        out, loss, accuracy = train_step_homogeneous(model,
                                                input_data,
                                                loss_compute_edges[key],
                                                loss_compute_edge_features[key],
                                                loss_compute_labels[key],
                                                criterion,
                                                optimizer,
                                                do_optim_step=False,
                                                edge_key=key)
        outs[key] = out
        accuracies.append(accuracy.cpu())
        losses.append(loss.cpu().detach().numpy())

    optimizer.step()

    mean_accuracy = np.mean(accuracies)
    mean_loss = np.mean(losses)

    return outs, mean_loss, mean_accuracy

def train_step(model,
        input_data,
        loss_compute_edges,
        loss_compute_edge_features,
        loss_compute_labels,
        criterion,
        optimizer):
    if model.is_heterogenous():
        return train_step_heterogeneous(model,
                                        input_data,
                                        loss_compute_edges,
                                        loss_compute_edge_features,
                                        loss_compute_labels,
                                        criterion,
                                        optimizer)
    else:
        return train_step_homogeneous(model,
                                      input_data,
                                      loss_compute_edges,
                                      loss_compute_edge_features,
                                      loss_compute_labels,
                                      criterion,
                                      optimizer)
'''
def train_step_recurrent(model,
        input_data,
        loss_compute_edges,
        loss_compute_edge_features,
        loss_compute_labels,
        criterion,
        optimizer,
        h, c):
    out, h, c = model(input_data.x, input_data.edge_index, loss_compute_edges, h, c)
    loss = criterion(out, loss_compute_labels)
    return loss, h, c
'''

def train(cfg,#TODO refactor to not take cfg
          num_epochs,
          node_featurizer,
          edge_featurizer,
          add_num_nodes=True,
          add_num_edges=True,
          use_edge_weights=False,
          num_labels_per_batch=1000,
          use_undirected_edges=False,
          logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info('Training...')
    writer = SummaryWriter(os.path.join(HydraConfig.get().runtime.output_dir, 'train'))
    model = make_model(cfg.model, node_featurizer, edge_featurizer)
    model.train()
    to_undirected = ToUndirected()

    input_data_path = '%s/train/'%cfg.collect_data_dir
    output_data_path = '%s/train/'%cfg.processed_dataset_dir

    dataset = SGMDataset(input_data_path,
                         output_data_path,
                         node_featurizer,
                         edge_featurizer,
                         include_labels=True,
                         add_num_nodes=add_num_nodes,
                         add_num_edges=add_num_edges,
                         num_workers=cfg.process_data_num_workers,
                         pre_transform=[to_undirected] if use_undirected_edges else None,
                         heterogenous=model.is_heterogenous(),
                         reverse_edges=model.reversed_edges)
    if cfg.no_cache:
        dataset.process()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    """
    splitter = RandomLinkSplit(num_val=0,
                               num_test=1.0,
                               is_undirected=True,
                               add_negative_train_samples=True,
                               neg_sampling_ratio=1.0,
                               disjoint_train_ratio=0.4)
    """
    if model.is_recurrent():
        loader = DataLoader(dataset, batch_size=5, shuffle=False)
    else:
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    #if model.is_recurrent():
    #    h, c = None, None

    # does not work for some reason...
    #criterion = torch.nn.CrossEntropyLoss(reduction='none')

    criterion = torch.nn.BCELoss(reduction='none')
    l2_loss = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        l2_loss_sum = 0
        accuracy_sum = 0
        epoch_loss = 0
        batch_count = 0 
        pbar = tqdm(loader)
        for data in pbar:
            batch_count+=1
            if not use_edge_weights:
                data.edge_weight = None
            #if not training:
            #    input_data, val_data, loss_compute_data = splitter(data.cuda())

            loss_inputs = sample_loss_edges(model, data, num_labels_per_batch, reversed_edges = model.reversed_edges)#, subsample=not model.include_transformer)
            loss_edges, loss_edge_features, loss_labels = loss_inputs

            data = data.cuda()

            #recurrent is not currenlty supported
            '''
            if model.is_recurrent():
                loss, h, c = train_step(model, data, loss_edges, loss_labels, criterion, optimizer, h, c, model.is_recurrent())
                total_loss+=loss/10
                epoch_loss+=loss.item()
                if (batch_count+1)%20 == 0:
                    h, c = None, None
                    total_loss.backward()
                    total_loss = 0
                    optimizer.step()
                    optimizer.zero_grad()
            '''

            out, loss, accuracy = train_step(
                                            model,
                                            data,
                                            loss_edges,
                                            loss_edge_features,
                                            loss_labels,
                                            criterion,
                                            optimizer)
            accuracy_sum += accuracy
            if type(out) is dict:
                for key in out:
                    loss_labels[key] = loss_labels[key] * (loss_labels[key] != 2).long()
                    l2_loss_sum+=l2_loss(out[key], loss_labels[key]).mean().cpu().item()
            else:
                loss_labels = loss_labels * (loss_labels != 2).long()
                l2_loss_sum+=l2_loss(out, loss_labels).mean().cpu().item()
            epoch_loss+=loss
            current_loss = epoch_loss/batch_count
            current_accuracy = accuracy_sum/batch_count
            l2_loss_avg = l2_loss_sum/batch_count
            pbar.set_description("Epoch %d | Avg Loss=%.4f | Avg l2 error = %.3f | Avg Accuracy=%.3f"%(epoch+1,
                                                                                                 current_loss,
                                                                                                 l2_loss_avg,
                                                                                                 current_accuracy))
        writer.add_scalar('avg_loss', current_loss, epoch + 1)
        writer.add_scalar('avg_Accuracy', current_accuracy, epoch + 1)

    logger.info('Train l2 error for %s model: %.3f'%(model.get_model_type(), l2_loss_avg))
    logger.info('Train avg accuracy for %s model: %.3f'%(model.get_model_type(), current_accuracy))

    model_path = create_path_to_model(
        cfg = cfg.model,
        node_featurizer = node_featurizer,
        edge_featurizer = edge_featurizer,
    )
    torch.save(model.state_dict(), model_path)

    return model

def test(cfg,#TODO refactor to not take cfg
         model,
         node_featurizer,
         edge_featurizer,
         add_num_nodes=True,
         add_num_edges=True,
         use_edge_weights=False,
         num_labels_per_batch=100,
         use_undirected_edges=False,
         logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info('Testing...')
    model.eval()
    to_undirected = ToUndirected()
    input_data_path = '%s/test/'%cfg.collect_data_dir
    output_data_path = '%s/test/'%cfg.processed_dataset_dir
    dataset = SGMDataset(input_data_path,
                         output_data_path,
                         node_featurizer,
                         edge_featurizer,
                         include_labels=True,
                         add_num_nodes=add_num_nodes,
                         add_num_edges=add_num_edges,
                         num_workers=cfg.process_data_num_workers,
                         pre_transform=[to_undirected] if use_undirected_edges else None,
                         heterogenous=model.is_heterogenous(),
                         reverse_edges=model.reversed_edges)
    if not model.is_recurrent():
        loader = DataLoader(dataset, batch_size=5, shuffle=False)
    else:
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    #if recurrent:
    #    h, c = None, None

    l2_loss = torch.nn.MSELoss()
    l2_loss_sum = 0
    accuracy_sum = 0
    batch_count = 0
    pbar = tqdm(loader)
    for data in pbar:
        if not use_edge_weights:
            data.edge_weight = None
        #if not training:
        #    input_data, val_data, loss_compute_data = splitter(data.cuda())
        test_edges, test_edge_features, test_labels = sample_loss_edges(model, data, num_labels_per_batch, reversed_edges = model.reversed_edges)
        data = data.cuda()

        if model.is_heterogenous():
            for key in test_edges:
                out = compute_output(model, data, test_edges[key], test_edge_features[key], edge_key=key)
                if len(out.size()) < 2 or len(test_labels[key].size()) < 2:
                    continue
                test_labels[key] = test_labels[key] * (test_labels[key] != 2).long()
                accuracy = (out.argmax(dim=1) == test_labels[key].argmax(dim=1)).float().mean()
                accuracy_sum+=accuracy.item()
                l2_loss_sum+=l2_loss(out, test_labels[key]).mean().cpu().item()
                batch_count+=1
        else:
            out = compute_output(model, data, test_edges, test_edge_features)

            test_labels = test_labels * (test_labels != 2).long()
            if len(out.size()) == 1:
                continue
            accuracy_sum+=(out.argmax(dim=1) == test_labels.argmax(dim=1)).float().mean().cpu().item()
            l2_loss_sum+=l2_loss(out, test_labels).mean().cpu().item()
            batch_count+=1

    logger.info('Test avg l2 error for %s model: %.3f'%(model.get_model_type(), l2_loss_sum/batch_count))
    logger.info('Test avg accuracy for %s model: %.3f\n'%(model.get_model_type(), accuracy_sum/batch_count))

    return accuracy_sum/batch_count
