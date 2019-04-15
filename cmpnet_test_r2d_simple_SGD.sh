python3 cmpnet_test.py --model_path ../CMPnet_res/r2d_simple/SGD/0.01/1/ \
--grad_step 1 --learning_rate 0.01 \
--memory_strength 0.5 --n_memories 1 \
--n_tasks 1 --device 3 --data_path /media/arclabdl1/HD1/Ahmed/rigid-body/dataset/ \
--start_epoch 1 --memory_type res --env_type r2d_simple --world_size 20 \
--total_input_size 2806 --AE_input_size 2800 --mlp_input_size 34 --output_size 3 \
--seen_N 10 --seen_NP 200 --seen_s 0 --seen_sp 4000 \
--unseen_N 10 --unseen_NP 200 --unseen_s 100 --unseen_sp 0 --opt SGD --train_path 1
# seen: 100, 200, 0, 4000
# unseen: 10, 2000, 100, 0
