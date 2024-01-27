cd ../src/

../../parallel -j10 --resume-failed --results ../Output/ER_4 --joblog ../joblog/ER_4 CUDA_VISIBLE_DEVICES=4 python run_exp.py  --seed {1} --train_ratio {2} --hidden_dim {3} --pooling_method sum --model_type {4} --dataset {5} ::: 0 10 20 30 40 50 60 70 80 90 ::: 0.7 0.9 ::: 64 32 ::: MPGNN GCN GCN_RW ::: ER-4 ER-5
