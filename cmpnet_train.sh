#python3 cmpnet_train.py --model_path ../CMPnet_res/c2d/ \
#--no_env 100 --no_motion_paths 4000 --grad_step 1 --learning_rate 0.01 \
#--num_epochs 1 --memory_strength 0.5 --n_memories 10000 \
#--n_tasks 1 --device 1 --freq_rehersal 100 --batch_rehersal 100 \
#--start_epoch 0 --data_path /media/arclabdl1/HD1/Ahmed/r-2d/ --world_size 20 --env_type c2d \
#--memory_type res
python2 cmpnet_train.py --model_path ../CMPnet_res/baxter/ \
--no_env 10 --no_motion_paths 500 --grad_step 1 --learning_rate 0.01 \
--num_epochs 10 --memory_strength 0.5 --n_memories 10000 \
--n_tasks 1 --device 0 --freq_rehersal 100 --batch_rehersal 100 \
--start_epoch 0 --data_path data/test/ --world_size 20 --env_type baxter \
--memory_type res --total_input_size 16067 --AE_input_size 16053 --mlp_input_size 74 --output_size 7 \
--dl1 1 --docker 0
