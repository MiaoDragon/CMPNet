cd ..
# End-2-End learning (randomly shuffle path)
python mpnet_train.py --model_path /media/arclabdl1/HD1/YLmiao/results/MPnet_res/home/ \
--no_env 1 --no_motion_paths 1800 --grad_step 1 --learning_rate 0.005 \
--num_epochs 100 --memory_strength 0.5 --n_memories 1 \
--n_tasks 1 --device 0 --freq_rehersal 100 --batch_rehersal 100 --batch_size 100 \
--start_epoch 0 --data_path /media/arclabdl1/HD1/YLmiao/data/home/ --world_size 20 --env_type home \
--memory_type res --total_input_size 150008 --AE_input_size 1 32 32 32 --mlp_input_size 78 --output_size 7
cd exp
