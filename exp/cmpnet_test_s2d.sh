cd ..
python3 cmpnet_test.py --model_path /media/arclabdl1/HD1/YLmiao/CMPnet_res/s2d/ \
--grad_step 1 --learning_rate 0.01 \
--memory_strength 0.5 --n_memories 10000 \
--n_tasks 1 --device 2 --data_path /media/arclabdl1/HD1/YLmiao/mpnet/data/simple/ \
--start_epoch 1 --memory_type res --env_type s2d --world_size 20 \
--total_input_size 2804 --AE_input_size 2800 --mlp_input_size 32 --output_size 2 \
--seen_N 100 --seen_NP 200 --seen_s 0 --seen_sp 4000 \
--unseen_N 10 --unseen_NP 2000 --unseen_s 100 --unseen_sp 0
# seen: 100, 200, 0, 4000
# unseen: 10, 2000, 100, 0
cd exp
