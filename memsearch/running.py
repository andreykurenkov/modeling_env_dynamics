import os
import functools
import numpy as np
import warnings
import itertools
import random
from multiprocessing import Pool, RLock
from tqdm import tqdm
from pathlib import Path
from tensorboardX import SummaryWriter
from memsearch.scene import make_scene_sampler_and_evolver
from memsearch.agents import make_agent
from memsearch.dataset import save_networkx_graph, make_featurizers
from memsearch.tasks import make_task, TaskType

def run(cfg,
        logger,
        output_dir,
        task_type = TaskType.PREDICT_LOC,
        num_steps = 10000,
        agent_type = 'random',
        save_graphs_path = None,
        save_images = False,
        for_data_collection = False,
        save_sgm_graphs = False,
        worker_num = 0):
    if not for_data_collection:
        writer = SummaryWriter(os.path.join(output_dir, agent_type))
    else:
        # Can ignore warning from networkx about pickle - remove if upgrading to networkx 3.0
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        writer = None
    images_dir = os.path.join(output_dir,'images')
    Path(images_dir).mkdir(exist_ok=True)

    scene_sampler, scene_evolver = make_scene_sampler_and_evolver(cfg.scene_gen)
    node_featurizer, edge_featurizer = make_featurizers(cfg.model, for_data_collection, cfg.task.num_steps)

    agents_cfg = cfg.agents
    if 'sgm' in agent_type or for_data_collection:
        model_config = cfg.model
    else:
        model_config = None
    agent = make_agent(agents_cfg, agent_type, task_type, node_featurizer, edge_featurizer, scene_sampler, scene_evolver, for_data_collection, model_config)
    sgm_agent = None
    if 'sgm' in agent_type:
        sgm_agent = agent
    elif for_data_collection:
        sgm_agent = agent.sgm_agent

    task = make_task(
        scene_sampler,
        scene_evolver,
        task_type,
        eps_per_scene=cfg.task.eps_per_scene)

    score_histories = []
    score_history = []
    sgm_graphs = []
    scene_num, scene_steps, scene_score, total_score, total_cost = 0, 0, 0, 0, 0

    current_scene = None
    pbar = tqdm(range(num_steps), position=worker_num, leave = not for_data_collection)
    
    for step in pbar:
        if task.scene is None or current_scene!=task.scene:
            scene_steps, scene_score = 0, 0
            if current_scene!=None:
                if writer is not None:
                    writer.add_scalar('average_score', average_score, scene_num)
                score_histories.append(np.array(score_history))
            else:
                query_nodes = task.reset()
            score_history = []
            scene_num+=1
            current_scene = task.scene
            agent.transition_to_new_scene(current_scene)

            if save_images and scene_num < 3:
                scene_sampler.current_priors_graph.save_png(f'{images_dir}/sampler_probs_{agent_type}_{scene_num}.png', colorize_edges=True)
            if save_images and scene_num < 3:
                scene_evolver.current_priors_graph.save_png(f'{images_dir}/evolver_probs_{agent_type}_{scene_num}.png', colorize_edges=True)

        if task_type == TaskType.PREDICT_ENV_DYNAMICS:
            prediction = agent.make_predictions(task_type, current_scene, query_nodes, top_k=cfg.task.top_k)
            obs, score, done, info = task.step(prediction, top_k=cfg.task.top_k, agent_type=agent_type)
        elif task_type == TaskType.FIND_OBJECT:
            prediction = agent.make_predictions(task_type, current_scene, query_nodes, top_k=cfg.task.top_k)
            obs, score, done, info = task.step(prediction, agent=agent, current_scene=current_scene, query_nodes=query_nodes, max_attempts=cfg.task.max_attempts, top_k=cfg.task.top_k)
            total_cost += info['acc_cost']

        else:
            prediction = agent.make_predictions(task_type, current_scene, query_nodes)
            obs, score, done, info = task.step(prediction)

        if for_data_collection:
            agent.mark_true_edges(current_scene.scene_graph)
            has_true_edge = False
            for edge in agent.sgm_agent.sgm_graph.get_edges():
                if edge.currently_true and edge.is_query_edge:
                    has_true_edge = True

            if not has_true_edge:
                logger.warning('None of the sgm edges considered are correct.')

            if save_sgm_graphs:
                save_path = save_graphs_path+f'/{worker_num}_{scene_num}_{scene_steps}.pickle'
                sgm_graphs.append(save_path)
                save_networkx_graph(save_path,
                                 sgm_agent.sgm_graph,
                                 node_featurizer,
                                 edge_featurizer)

            else:
                sgm_graphs.append(sgm_agent.sgm_graph.copy())

            agent.remove_hyp_query_edges()

        if save_images and step % 10 == 0 and scene_num < 3:
            current_scene.scene_graph.save_pngs(f'{images_dir}/scene_{agent_type}_{scene_num}_{scene_steps}.png')
            if 'sgm' in agent_type:
                agent.sgm_graph.save_pngs(f'{images_dir}/sgm_{scene_num}_{scene_steps}.png')
            elif for_data_collection:
                agent.sgm_agent.sgm_graph.save_pngs(f'{images_dir}/sgm_{scene_num}_{scene_steps}.png')

        agent.receive_observation(obs)

        scene_steps+=1
        scene_score+=float(score)
        total_score+=float(score)
        average_cost=total_cost/(step+1)
        score_history.append(float(score))
        average_score=total_score/(step+1)
        scene_average_score=scene_score/(scene_steps)
        
        #short agent type to make things line up nicely
        if agent_type == 'sgm':
            short_agent_type = 'sgm   '
        else:
            short_agent_type = agent_type[:6]
        if for_data_collection:
            pbar.set_description(("Agent %s | Step %d | Scene %d with %d Nodes | sgm has %d nodes & %.0f edges | Scene Accuracy=%.2f | "+\
                                  "Overall Accuracy=%.2f")%(short_agent_type,
                                                            step+1,
                                                            scene_num,
                                                            len(current_scene.scene_graph.nodes),
                                                            float(len(sgm_agent.sgm_graph.nodes)),
                                                            float(len(sgm_agent.sgm_graph.get_edges())),
                                                            scene_average_score,
                                                            average_score))
        elif task_type == TaskType.FIND_OBJECT:
            pbar.set_description(("Agent %s | Overall Accuracy=%.2f | Average Cost=%.2f")%(short_agent_type, average_score, average_cost))
        else:
            pbar.set_description(("Agent %s | Overall Accuracy=%.2f")%(short_agent_type, average_score))

        if done:
            query_nodes = task.reset()
            agent.step()
    pbar.close()
    if for_data_collection:
        if worker_num == 0:
            node_featurizer.save_text_embedding_dict()
        return sgm_graphs
    else:
        if 'sgm' in agent_type:
            node_featurizer.save_text_embedding_dict()
        return score_histories

def collect_data(cfg,
                 logger,
                 output_dir,
                 num_steps = 10000,
                 agent_type = 'random',
                 save_graphs_path = None,
                 save_sgm_graphs = True):
    task_type = TaskType.PREDICT_LOC
    if cfg.collect_data_num_workers == 1:
        data = run(cfg, 
            logger,
            output_dir,
            task_type,
            num_steps,
            agent_type,
            save_graphs_path,
            cfg.save_images,
            for_data_collection = True,
            save_sgm_graphs = save_sgm_graphs)
    else:
        assert num_steps % cfg.collect_data_num_workers == 0
        num_steps = int(num_steps / cfg.collect_data_num_workers)
        run_in_parallel = functools.partial(run, cfg, logger, output_dir, task_type, num_steps, 
                                            agent_type, save_graphs_path, cfg.save_images, True, 
                                            save_sgm_graphs)
        tqdm.set_lock(RLock())
        with Pool(processes = cfg.collect_data_num_workers) as pool:
            data_list = pool.map(run_in_parallel, list(range(cfg.collect_data_num_workers)))
        # flatten into one list
        data = list(itertools.chain.from_iterable(data_list))
    return data

def eval_agent(cfg,
               logger,
               output_dir,
               task_type = TaskType.PREDICT_LOC,
               agent_type = 'random'):
    worker_num = cfg.agents.agent_types.index(agent_type)
    random.seed(0)
    return run(cfg, 
               logger,
               output_dir,
               task_type=task_type,
               num_steps=cfg.task.num_steps,
               agent_type=agent_type,
               save_images=cfg.save_images,
               for_data_collection=False,
               worker_num=worker_num)
