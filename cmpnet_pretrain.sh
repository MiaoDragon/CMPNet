python3 cmpnet_train.py --model_path ../CMPnet_res/s2d/ \
--no_env 4 --no_motion_paths 200 --grad_step 1 --learning_rate 0.01 \
--num_epochs 1 --memory_strength 0.5 --n_memories 500 \
--n_tasks 1 --device 0 --freq_rehersal 100 --batch_rehersal 100 \
--start_epoch 0 --data_path ../data/simple/ --world_size 20 --env_type s2d \
--memory_type res
