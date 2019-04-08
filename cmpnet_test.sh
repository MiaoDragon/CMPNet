#python3 cmpnet_test.py --model_path ../CMPnet_res/c2d/ \
#--no_env 100 --no_motion_paths 4000 --grad_step 1 --learning_rate 0.01 \
#--memory_strength 0.5 --n_memories 10000 \
#--n_tasks 1 --device 1 --data_path /media/arclabdl1/HD1/Ahmed/r-2d/ \
#--start_epoch 1 --memory_type res --env_type c2d --world_size 20
python cmpnet_test.py --model_path CMPnet_res/baxter_test_small_m20000_ms50_b0_f0/ \
--grad_step 1 --learning_rate 0.01 \
--memory_strength 0.5 --n_memories 10000 \
--n_tasks 1 --device 0 --data_path ../data/test_100/ \
--start_epoch 1 --memory_type res --env_type baxter --world_size 20 \
--total_input_size 16067 --AE_input_size 16053 --mlp_input_size 74 --output_size 7 \
--seen_N 10 --seen_NP 200 --seen_s 0 --seen_sp 4000 \
--unseen_N 10 --unseen_NP 200 --unseen_s 100 --unseen_sp 0

