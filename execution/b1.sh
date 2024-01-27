cd ../src/

../../parallel -j10 --resume-failed --results ../Output/PROTEINS_1 --joblog ../joblog/PROTEINS_1 CUDA_VISIBLE_DEVICES=1 python run_exp.py --dataset PROTEINS  --batch_size 128 --max_epoch 50 --seed {1} --train_ratio {2} --hidden_dim {3} --pooling_method sum --model_type {4} ::: 0 10 20 30 40 50 60 70 80 90 ::: 0.7 0.9 ::: 128 64 4 ::: MPGNN GCN GCN_RW
