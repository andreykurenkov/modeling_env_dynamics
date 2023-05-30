# Modeling Dynamic Environments with Scene Graph Memory (ICML 2023)

## Abstract

Embodied AI agents that search for objects in dynamic environments such as households often need to make efficient decisions by predicting object locations based on partial information. We pose this problem as a new type of link prediction problem: \textbf{link prediction on partially observable dynamic graphs}. Our graph is a representation of a scene in which rooms and objects are nodes, and their relationship (e.g., containment) is encoded in the edges; only parts of the changing graph are known to the agent at each step. This partial observability poses a challenge to existing link prediction approaches, which we address. We propose a novel representation of the agent’s accumulated set of observations into a state representation called a Scene Graph Memory (SGM), as well as a neural net architecture called a Node Edge Predictor (NEP) that extracts information from the SGM to search efficiently. We evaluate our method in the Dynamic House Simulator, a new benchmark that creates diverse dynamic graphs following the semantic patterns typically seen at homes, and show that NEP can be trained to predict the locations of objects in a variety of environments with diverse object movement dynamics, outperforming baselines both in terms of new scene adaptability and overall accuracy.

## Installation

Note: You may use conda instead of mamba

```bash
git clone git@github.com:andreykurenkov/memory_object_search.git
cd memory_object_search
conda env create -f environment.yml
conda activate mos
python -m spacy download en_core_web_sm
chmod +x install.sh
./install.sh
```

# Updating

```bash
conda env update --file environment.yml --prune
```

# Running

```python
# Generate prior graphs
python scripts/gen_prior_graphs.py

# Generate the data
python scripts/collect_data.py 

# Train the model 
python scripts/train.py --multirun model=mlp,gcn,heat

# Evaluate the model
python scripts/eval.py 
```
# Specific experiments
To run specific experiments, just append experiment=<experiment_name> to the above commands after the python file name. EG:
```python
python scripts/gen_prior_graphs.py experiment=example
python scripts/train.py --multirun experiment=example model=mlp,gcn,heat
python scripts/eval.py
```
# Running simulations with iGridson
iGridson simulations supports two modes: headless and render. In headless mode, the agent will navigate through the apartment environment without producing or saving any visualizations. In render mode, the visualization will be showed in a window and each frame will be saved to outputs/igridson_simulations/<experiment_name>. This parameter can be changing the `viz_mode` parameter in the config file for configs/experiment/simulate_agents.

Next, copy any trained models you want the simulator to use to ```outputs/simulate_agents/models```.

Finally, run
```python
python scripts/igridson_video.py experiment=simulate_agents
```

# Sweeping multiple parameters
```python
python scripts/eval.py --multirun +eval=predict_location ++changes_per_step=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 ++obsevation_prob=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0
```

# Generating config files for experiment(s)

Use ``` python scripts/yaml_gen.py``` to generate config files for experiments. The scripts expects a text file as input. Either enter strings such as pl_l_d_r_l_za_npp (one on each line), or tab-spaced variables such as 
predict_dynamics	small	detailed	coarse	large	none	None
,also one on each line 

Note that the tab spaced variables are what you would get if you simply copy multiple rows from a spreadsheet.

The script by default expects a text file at data/iclr_experiments.txt containing formatted strings (such as pl_l_d_r_l_za_npp)

To use your own text file, use
```
python scripts/generate_experiment_configs.py --text_file_path <PATH_TO_TEXT_FILE>
```

To use tab separated variable names, use
```
python scripts/generate_experiment_configs.py --use_tab_separated_vars
```

To print out the formatted strings for generated config files, use
```
python scripts/generate_experiment_configs.py --print_formatted_strings
```

# Reinforcement Learning (Experimental)
Some scripts for training RL agents is provided under rl_scripts/. Note that this is an experimental feature that hasn't been thoroughly tested and did not give exceptional results in our preliminary tests. To run a basic training script, run:
```python
python rl_scripts/train.py experiment=simulate_agents
```

Also note that it needs a configuration file similar to simulate_agents as it uses a headless iGridson environment.

## Cite

```bibtex
@article{kurenkov2023modeling,
  title={Modeling Dynamic Environments with Scene Graph Memory},
  author={Kurenkov, Andrey and Lingelbach, Michael and Agarwal, Tanmay and Li, Chengshu and Jin, Emily, and Fei-Fei, Li and Wu, Jiajun and Savarese, Silvio, and Mart{\'i}n-Mart{\'i}n, Roberto},
  booktitle={International Conference on Machine Learning},
  year={2023},
  organization={PMLR}
}
```
