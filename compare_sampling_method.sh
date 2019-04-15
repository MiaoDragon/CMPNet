# state space has 3DOF
# input of MLP is thus 3*2+28=34
python3 cmpnet_compare_sample.py --model_path ../CMPnet_res/s2d/ \
--no_env 2 --no_motion_paths 10 --grad_step 1 --learning_rate 0.01 \
--num_epochs 1 --memory_strength 0.5 --n_memories 10 \
--n_tasks 1 --device 0 --test_frequency 10 \
--start_epoch 0 --data_path /media/arclabdl1/HD1/Ahmed/rigid-body/dataset/ --world_size 20 --env_type s2d \
--memory_type res --total_input_size 2804 --AE_input_size 2800 --mlp_input_size 32 --output_size 2 --train_path 1
