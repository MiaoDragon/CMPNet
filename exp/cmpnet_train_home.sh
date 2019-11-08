cd ..
# End-2-End learning (randomly shuffle path)
python cmpnet_train.py --model_path ../CMPnet_res/home/ \
--no_env 1 --no_motion_paths 2700 --grad_step 1 --learning_rate 0.001 \
--num_epochs 10 --memory_strength 0.5 --n_memories 10000 \
--n_tasks 1 --device 0 --freq_rehersal 100 --batch_rehersal 100 \
--start_epoch 0 --data_path /media/arclabdl1/HD1/YLmiao/data/home/ --world_size 20 --env_type home \
--memory_type res --total_input_size 150008 --AE_input_size 149994 --mlp_input_size 142 --output_size 7
cd exp
