cd ../src/

../../parallel -j10 --resume-failed --results ../Output/ER_2 --joblog ../joblog/ER_2 CUDA_VISIBLE_DEVICES=2 python run_exp.py  --seed {1} --train_ratio {2} --hidden_dim {3} --pooling_method mean --model_type {4} --dataset {5} ::: 0 10 20 30 40 50 60 70 80 90 ::: 0.1 0.3 0.5 0.7 0.9 ::: 256 8 4 ::: MPGNN GCN GCN_RW ::: ER-4 ER-5
