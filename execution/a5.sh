cd ../src/

../../parallel -j10 --resume-failed --results ../Output/SBM1_5 --joblog ../joblog/SBM1_5 CUDA_VISIBLE_DEVICES=5 python run_exp.py --load_only  --seed {1} --train_ratio {2} --hidden_dim {3} --pooling_method mean --model_type {4} --dataset SBM-1 ::: 0 10 20 30 40 50 60 70 80 90 ::: 0.7 0.9 ::: 256 128 64 32 16 8 4 ::: MPGNN GCN GCN_RW
