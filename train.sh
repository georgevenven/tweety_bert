#!/bin/bash

python3 TweetyBERT.py --experiment_name "Song_Detector_Pretrain" \
                      --continue_training False \
                      --train_dir "/media/george-vengrovski/disk2/training_song_detector/pretrain_dataset_train" \
                      --test_dir "/media/george-vengrovski/disk2/training_song_detector/pretrain_dataset_test" \
                      --batch_size 48 \
                      --d_transformer 12 \
                      --nhead_transformer 2 \
                      --num_freq_bins 196 \
                      --dropout 0.2 \
                      --dim_feedforward 24 \
                      --transformer_layers 3 \
                      --m 1000 \
                      --p 0.01 \
                      --alpha 1 \
                      --pos_enc_type "relative" \
                      --pitch_shift True \
                      --learning_rate 3e-4 \
                      --max_steps 1e4 \
                      --eval_interval 100 \
                      --save_interval 500 \
                      --context 1000 \
                      --weight_decay 0.0 \
                      --early_stopping True \
                      --patience 8 \
                      --trailing_avg_window 200 \
                      --num_ground_truth_labels 50
