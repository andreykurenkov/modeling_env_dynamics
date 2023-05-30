export experiment=$1
echo "Submitting sbatch job for experiment $experiment"
sbatch --output=/cvgl2/u/andreyk/projects/memory_object_search/outputs/slurm/$experiment.out --account viscam --error=/cvgl2/u/andreyk/projects/memory_object_search/outputs/slurm/$experiment.err --job-name=$experiment slurm_run_experiment.sbatch
