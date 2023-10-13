# GNN_MF_GE
Generalization Error of Graph Neural Network in the Mean-field Regime.

--------------------------------------------------------------------------------

## Environment Setup
The codebase is implemented in Python 3.7. package versions used for development are below.
```
networkx                        2.6.3
numpy                           1.20.3
scipy                           1.7.1
argparse                        1.1.0
torch                           1.10.1
pyg                             2.0.3
```

## Folder structure
- ./execution/ stores files that can be executed to generate outputs. For vast number of experiments, we use [GNU parallel](https://www.gnu.org/software/parallel/), which can be downloaded in command line and make it executable via:
```
wget http://git.savannah.gnu.org/cgit/parallel.git/plain/src/parallel
chmod 755 ./parallel
```

- ./joblog/ stores job logs from parallel. 
You might need to create it by 
```
mkdir joblog
```

- ./Output/ stores raw outputs (ignored by Git) from parallel.
You might need to create it by 
```
mkdir Output
```

- ./data/ stores processed data sets.

- ./result_arrays/ stores results for different data sets. Each data set has a separate subfolder.

- ./exp/ stores trained models and logs.

## Reproduce results
First, get into the ./execution/ folder:
```
cd execution
```
To reproduce the results to be executed on GPU-0 for machine a.
```
bash a0.sh
```

Note that if you are operating on CPU, you may delete the commands ``CUDA_VISIBLE_DEVICES=xx". You can also set you own number of parallel jobs, not necessarily following the j numbers in the .sh files, or use other GPU numbers.

## Direct execution with training files

Below are various options to try:

Creating a GCN model using the mean-readout function for the ER-4 data set.
```
python ./run_exp.py --model_type GCN --dataset ER-4 --pooling_method mean
```
Creating an MPGNN model using the sum-readout function for the PROTEINS data set and run for 50 epochs.
```
python ./run_exp.py --model_type MPGNN --max_epochs 50 --dataset PROTEINS --pooling_method sum
```
--------------------------------------------------------------------------------