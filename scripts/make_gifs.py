from PIL import Image
import os
from pygifsicle import optimize
# filepaths
types=['evolver_probs_psg_gcn', 
       'evolver_probs_psg_mlp', 
       'evolver_probs_psg_heat', 
       'edge_age_psg_mlp',
       'edge_age_psg_gcn',
       'edge_age_psg_heat',
       'state_psg_mlp',
       'state_psg_gcn',
       'state_psg_heat',
       'node_moves_psg_mlp', 
       'node_moves_psg_gcn', 
       'node_moves_psg_heat', 
       'priors',
#       'psg',
       'times_observed',
       'freq_true_MLP',
       'freq_true_GCN',
       'freq_true_HEAT',
       'times_true',
       'from_priors',
       'from_observed',
       'last_outputs_MLP',
       'last_outputs_GCN',
       'last_outputs_HEAT']
         
for t in types:
    print(t)
    if 'evolver_' in t:
        paths = sorted(['graphs/'+f for f in os.listdir('graphs') if f.startswith(t)],
                        key=lambda x: int(x.split('_')[-1].replace('.png','')))
    elif 'edge_age' in t or 'node_moves' in t or 'state' in t:
        paths = sorted(['graphs/'+f for f in os.listdir('graphs') if f.startswith('scene_'+t)],
                        key=lambda x: int(x.split('_')[-2])*500 + int(x.split('_')[-1].replace('.png','')))
    else:
        paths = sorted(['graphs/'+f for f in os.listdir('graphs') if f.startswith('psg_'+t)],
                        key=lambda x: int(x.split('_')[-2])*500 + int(x.split('_')[-1].replace('.png','')))
    imgs = (Image.open(f) for f in paths)
    img = next(imgs)  # extract first image from iterator
    fp_out = 'gifs/'+t+".gif"
    img.save(fp=fp_out, format='GIF', append_images=imgs,
	     save_all=True, duration=40000/len(paths), loop=1)
    optimize(fp_out)
