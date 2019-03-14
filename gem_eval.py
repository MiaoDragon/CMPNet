import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from torch.autograd import Variable
import math
import time
from plan_general import *

def eval_tasks(mpNet, test_data, filename, IsInCollision, normalize_func = lambda x:x, unnormalize_func=lambda x: x):
    obc, obs, paths, path_lengths = test_data
    obs = torch.from_numpy(obs)
    fes_env = []   # list of list
    valid_env = []
    #for i in range(0,1):
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
            fp = 0 # indicator for feasibility
            print ("step: i="+str(i)+" j="+str(j))
            #for j in range(0,2):
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
                MAX_NEURAL_REPLAN = 11
                for t in range(MAX_NEURAL_REPLAN):
                # adaptive step size on replanning attempts
                    if (t == 2):
                        step_sz = 0.04
                    elif (t == 3):
                        step_sz = 0.03
                    elif (t > 3):
                        step_sz = 0.02
                    path = neural_replan(mpNet, path, obc[i], obs[i], IsInCollision, \
                                         normalize_func, unnormalize_func, t==0, step_sz=step_sz)
                    path = lvc(path, obc[i], IsInCollision, step_sz=step_sz)
                    if feasibility_check(path, obc[i], IsInCollision, step_sz=0.01):
                        fp = 1
                        print('feasible, ok!')
                        break
            if fp:
                # only for successful paths
                time1 = time.time() - time0
                time_path.append(time1)
                print('test time: %f' % (time1))
            fes_path.append(fp)
        time_env.append(time_path)
        time_total += time_path
        print('average test time up to now: %f' % (np.mean(time_total)))
        fes_env.append(fes_path)
        valid_env.append(valid_path)
        print('accuracy up to now: %f' % (np.sum(fes_env) / np.sum(valid_env)))
    pickle.dump(time_env, open(filename, "wb" ))
        #print(fp/tp)
    return np.array(fes_env), np.array(valid_env)
