cd ../src/

../../parallel -j10 --resume-failed --results ../Output/SBM3_7 --joblog ../joblog/SBM3_7 CUDA_VISIBLE_DEVICES=7 python run_exp.py  --seed {1} --train_ratio {2} --hidden_dim {3} --pooling_method mean --model_type {4} --dataset SBM-3 ::: 0 10 20 30 40 50 60 70 80 90 ::: 0.1 0.3 0.5 0.7 0.9 ::: 256 128 64 32 16 8 4 ::: MPGNN GCN GCN_RW

../../parallel -j10 --resume-failed --results ../Output/PROTEINS_7 --joblog ../joblog/PROTEINS_7 CUDA_VISIBLE_DEVICES=7 python run_exp.py --dataset PROTEINS  --batch_size 128 --max_epoch 50 --seed {1} --train_ratio {2} --hidden_dim {3} --pooling_method mean --model_type {4} ::: 0 10 20 30 40 50 60 70 80 90 ::: 0.1 0.3 0.5 0.7 0.9 ::: 32 ::: MPGNN GCN GCN_RW
