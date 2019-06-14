cd ..
python3 cmpnet_train.py --model_path ../CMPnet_res/r3d/ \
--no_env 10 --no_motion_paths 400 --grad_step 1 --learning_rate 0.01 \
--num_epochs 1 --memory_strength 0.5 --n_memories 10000 \
--n_tasks 1 --device 3 --freq_rehersal 100 --batch_rehersal 100 \
--start_epoch 0 --data_path /media/arclabdl1/HD1/Ahmed/r-3d/dataset2/ --world_size 20 --env_type r3d \
--memory_type res --total_input_size 6006 --AE_input_size 6000 --mlp_input_size 66 --output_size 3
cd exp
