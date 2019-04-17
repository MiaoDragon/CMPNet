# state space has 3DOF
# input of MLP is thus 3*2+28=34
python3 cmpnet_compare_sample.py --model_path ../CMPnet_res/s2d/sample_exp/loss/ \
--no_env 10 --no_motion_paths 4000 --grad_step 1 --learning_rate 0.01 \
--num_epochs 1 --memory_strength 0.5 --n_memories 500 \
--n_tasks 1 --device 2 --test_frequency 1000 \
--start_epoch 0 --data_path /media/arclabdl1/HD1/YLmiao/mpnet/data/simple/ --world_size 20 --env_type s2d \
--memory_type prio_loss --total_input_size 2804 --AE_input_size 2800 --mlp_input_size 32 --output_size 2 --train_path 1 \
--seen_N 10 --seen_NP 50 --seen_s 0 --seen_sp 4000 \
--unseen_N 10 --unseen_NP 50 --unseen_s 100 --unseen_sp 0

#python3 cmpnet_compare_sample.py --model_path ../CMPnet_res/s2d/sample_exp/ \
#--no_env 2 --no_motion_paths 5 --grad_step 1 --learning_rate 0.01 \
#--num_epochs 1 --memory_strength 0.5 --n_memories 4 \
#--n_tasks 1 --device 0 --test_frequency 4 \
#--start_epoch 0 --data_path /media/arclabdl1/HD1/YLmiao/mpnet/data/simple/ --world_size 20 --env_type s2d \
#--memory_type prio_loss --total_input_size 2804 --AE_input_size 2800 --mlp_input_size 32 --output_size 2 --train_path 1 \
#--seen_N 2 --seen_NP 5 --seen_s 0 --seen_sp 4000 \
#--unseen_N 2 --unseen_NP 5 --unseen_s 100 --unseen_sp 0
