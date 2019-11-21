cd ..
python cmpnet_test.py --model_path /media/arclabdl1/HD1/YLmiao/results/CMPnet_res/home_mlp4/ \
--grad_step 1 --learning_rate 0.001 \
--memory_strength 0.5 --n_memories 1 \
--n_tasks 1 --device 3 --data_path /media/arclabdl1/HD1/YLmiao/data/home/ \
--start_epoch 100 --memory_type res --env_type home_mlp4 --world_size 20 \
--total_input_size 150008 --AE_input_size 1 32 32 32 --mlp_input_size 78 --output_size 7 \
--seen_N 1 --seen_NP 1800 --seen_s 0 --seen_sp 200 \
--unseen_N 0 --unseen_NP 0 --unseen_s 0 --unseen_sp 0
# seen: 100, 200, 0, 4000
# unseen: 10, 2000, 100, 0
cd exp
