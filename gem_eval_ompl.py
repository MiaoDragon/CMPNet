import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from torch.autograd import Variable
import math
import time
from plan_general_ompl import *

def eval_tasks(mpNet, test_data, filename, IsInCollision, normalize_func = lambda x:x, unnormalize_func=lambda x: x, time_flag=False):
    obc, obs, paths, path_lengths = test_data
    obs = obs.astype(np.float32)
    obs = torch.from_numpy(obs)
    fes_env = []   # list of list
    valid_env = []
    time_env = []
    time_total = []
    for i in range(len(paths)):
        time_path = []
        fes_path = []   # 1 for feasible, 0 for not feasible
        valid_path = []      # if the feasibility is valid or not
        # save paths to different files, indicated by i
        # feasible paths for each env
        for j in range(len(paths[0])):
            time0 = time.time()
            time_norm = 0.
            fp = 0 # indicator for feasibility
            print ("step: i="+str(i)+" j="+str(j))
            p1_ind=0
            p2_ind=0
            p_ind=0
            if path_lengths[i][j]==0:
                # invalid, feasible = 0, and path count = 0
                fp = 0
                valid_path.append(0)
            if path_lengths[i][j]>0:
                fp = 0
                valid_path.append(1)
                path = [torch.from_numpy(paths[i][j][0]).type(torch.FloatTensor),\
                        torch.from_numpy(paths[i][j][path_lengths[i][j]-1]).type(torch.FloatTensor)]

                step_sz = DEFAULT_STEP
                #MAX_NEURAL_REPLAN = 11
                MAX_NEURAL_REPLAN = 51
                for t in range(MAX_NEURAL_REPLAN):
                # adaptive step size on replanning attempts
                # 1.2, 0.5, 0.1 are for simple env
                # 0.04, 0.03, 0.02 are for home env
                    if (t == 2):
                        #step_sz = 1.2
                        step_sz = 0.04
                    elif (t == 3):
                        #step_sz = 0.5
                        step_sz = 0.03
                    elif (t > 3):
                        #step_sz = 0.1
                        step_sz = 0.02
                    if time_flag:
                        path, time_norm = neural_replan(mpNet, path, obc[i], obs[i], IsInCollision, \
                                            normalize_func, unnormalize_func, t==0, step_sz=step_sz, time_flag=time_flag)
                    else:
                        path = neural_replan(mpNet, path, obc[i], obs[i], IsInCollision, \
                                            normalize_func, unnormalize_func, t==0, step_sz=step_sz, time_flag=time_flag)
                    print('after neural replan %d:' % (t))
                    #print(path)

                    #path_vis = [p.numpy() for p in path]
                    #path_vis = np.array(path_vis)
                    #np.savetxt('path_%d_replan_%d.txt' % (j, t), path_vis, fmt='%f')
                    # for several paths at the beginning, don't do this
                    if t > (MAX_NEURAL_REPLAN * 10):
                        path = dist_lvc(path, obc[i], IsInCollision, step_sz=step_sz)
                        #path_vis = [p.numpy() for p in path]
                        #path_vis = np.array(path_vis)
                        #np.savetxt('path_%d_replan_%d_reordered.txt' % (j, t), path_vis, fmt='%f')
                    path = lvc(path, obc[i], IsInCollision, step_sz=step_sz)
                    #path_vis = [p.numpy() for p in path]
                    #path_vis = np.array(path_vis)
                    #np.savetxt('path_%d_replan_%d_lvc.txt' % (j, t), path_vis, fmt='%f')
                    #print('after lvc:')
                    #print(path)
                    if feasibility_check(path, obc[i], IsInCollision, step_sz=0.01):
                        fp = 1
                        print('feasible, ok!')
                        break
            if fp:
                # only for successful paths
                time1 = time.time() - time0
                time1 -= time_norm
                time_path.append(time1)
                print('test time: %f' % (time1))
            # write the path
            #print('planned path:')
            #print(path)
            path = [p.numpy() for p in path]
            path = np.array(path)
            if fp:
                np.savetxt('path_%d_fes.txt' % (j), path, fmt='%f')
            else:
                np.savetxt('path_%d_nfes.txt' % (j), path, fmt='%f')

            fes_path.append(fp)
            print('env %d accuracy up to now: %f' % (i, (float(np.sum(fes_path))/ np.sum(valid_path))))
        time_env.append(time_path)
        time_total += time_path
        print('average test time up to now: %f' % (np.mean(time_total)))
        fes_env.append(fes_path)
        valid_env.append(valid_path)
        print('accuracy up to now: %f' % (float(np.sum(fes_env)) / np.sum(valid_env)))
    if filename is not None:
        pickle.dump(time_env, open(filename, "wb" ))
        #print(fp/tp)
    return np.array(fes_env), np.array(valid_env)
