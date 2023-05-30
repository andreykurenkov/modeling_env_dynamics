# Generate the data
python scripts/collect_data.py experiment=$1

# Train the models
python scripts/train.py model=mlp experiment=$1 
python scripts/train.py model=gcn experiment=$1 
python scripts/train.py model=heat experiment=$1
python scripts/train.py model=hgt experiment=$1
python scripts/train.py model=han experiment=$1
#python scripts/train.py model=hgcn experiment=$1

# Evaluate the model
python scripts/eval.py experiment=$1
