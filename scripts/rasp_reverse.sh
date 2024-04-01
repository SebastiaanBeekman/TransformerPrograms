#!/bin/bash 

SEED=0

for SEED in 0 1 2 3 4
do
     python ../src/run_old.py \
          --dataset "reverse" \
          --dataset_size 20000 \
          --seed "${SEED}" \
          --d_var 8 \
          --n_heads_cat 4 \
          --n_heads_num 4 \
          --n_cat_mlps 1 \
          --n_num_mlps 1 \
          --n_layers 3 \
          --one_hot_embed \
          --count_only \
          --save \
          --save_code \
          --device "cpu" \
          --output_dir "output/rasp/reverse/s${SEED}";
done          

# --dataset "reverse" --dataset_size 20000 --seed 0 --d_var 8 --n_heads_cat 4 --n_heads_num 4 --n_cat_mlps 1 --n_num_mlps 1 --n_layers 3 --one_hot_embed --count_only --save --save_code --device "cpu" --output_dir "output/rasp/reverse/s0"