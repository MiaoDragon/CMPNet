#python3 cmpnet_test.py --model_path ../CMPnet_res/c2d/ \
#--no_env 100 --no_motion_paths 4000 --grad_step 1 --learning_rate 0.01 \
#--memory_strength 0.5 --n_memories 10000 \
#--n_tasks 1 --device 1 --data_path /media/arclabdl1/HD1/Ahmed/r-2d/ \
#--start_epoch 1 --memory_type res --env_type c2d --world_size 20
python3 cmpnet_test.py --model_path ../CMPnet_res/s2d/ \
--no_env 100 --no_motion_paths 500 --grad_step 1 --learning_rate 0.01 \
--memory_strength 0.5 --n_memories 10000 \
--n_tasks 1 --device 1 --data_path ../data/simple/ \
--start_epoch 1 --memory_type res --env_type s2d --world_size 20 \
--total_input_size 2804 --AE_input_size 2800 --mlp_input_size 32
