#SBATCH -p gpu_requeue  ## partition: must be requesting GPUs to use this
#SBATCH -n 1  ## Number of CPU cores
#SBATCH --gres=gpu:1  ## Number of GPU cores requested
#SBATCH --constraint=cuda-7.5  ## make sure the right version of CUDA present
#SBATCH -t 0-2:00  ## request one hour (overkill for installations)
#SBATCH --mem=2500  ## request 2.5 GB memory
#SBATCH -o %j.out  ## stdout will be written to [job number].out
#SBATCH -e %j.err  ## any error messages will be written to [job number].err

# load required modules
module load Anaconda3/4.3.0-fasrc01 gcc/4.9.3-fasrc01 cuda/7.5-fasrc02 cudnn/7.0-fasrc02 tensorflow/1.0.0-fasrc04

# load the required virtual environment
source activate /n/garner_lab/lab/virtual_envs/deepscope

python3 /n/denic_lab/Users/nweir/statoil_odyssey/vgg19_1.py
