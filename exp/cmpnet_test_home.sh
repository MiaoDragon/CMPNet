cd ..
python3 cmpnet_test.py --model_path ../MPnet_res/home/ \
--grad_step 1 --learning_rate 0.001 \
--memory_strength 0.5 --n_memories 1 \
--n_tasks 1 --device 0 --data_path /media/arclabdl1/HD1/YLmiao/data/home/ \
--start_epoch 10 --memory_type res --env_type home --world_size 20 \
--total_input_size 150014 --AE_input_size 150000 --mlp_input_size 142 --output_size 7 \
--seen_N 1 --seen_NP 5 --seen_s 0 --seen_sp 4000 \
--unseen_N 0 --unseen_NP 0 --unseen_s 0 --unseen_sp 0
# seen: 100, 200, 0, 4000
# unseen: 10, 2000, 100, 0
cd exp
