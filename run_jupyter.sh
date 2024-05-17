!/bin/bash
#
#SBATCH --mail-user=helentian@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/helentian/slurm/slurm_out/%j.%N.stdout
#SBATCH --error=/home/helentian/slurm/slurm_out/%j.%N.stderr
#SBATCH --partition=pascal
#SBATCH --account=helentian
#SBATCH --job-name=run_P3unsupervised_jupyter_notebook
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=1000
#SBATCH --time=02:00:00

module load python/3.8
module load cuda/11.0

# Activate your Python virtual environment if you have one
source /usr/bin/activate

# Navigate to the directory containing your notebook
cd ./P3_unsupervised.ipynb

# Start Jupyter Notebook
jupyter-notebook --no-browser --ip=0.0.0.0 --port=8888
