import torch
import numpy as np
from utility import *
import time
DEFAULT_STEP = 2.

from os.path import join
from ompl import base as ob
from ompl import app as oa
from ompl import geometric as og

def allocatePlanner(si, plannerType):
    if plannerType.lower() == "bfmtstar":
        return og.BFMT(si)
    elif plannerType.lower() == "bitstar":
        return og.BITstar(si)
    elif plannerType.lower() == "fmtstar":
        return og.FMT(si)
    elif plannerType.lower() == "informedrrtstar":
        return og.InformedRRTstar(si)
    elif plannerType.lower() == "prmstar":
        return og.PRMstar(si)
    elif plannerType.lower() == "rrtstar":
        return og.RRTstar(si)
    elif plannerType.lower() == "sorrtstar":
        return og.SORRTstar(si)
    else:
        ou.OMPL_ERROR("Planner-type is not implemented in allocation function.")

ompl_app_root = "/home/arclabdl1/ompl/omplapp-1.4.2-Source/"
ompl_resources_dir = join(ompl_app_root, 'resources/3D')

setup = oa.SE3RigidBodyPlanning()
setup.setRobotMesh(join(ompl_resources_dir, 'Home_robot.dae'))
setup.setEnvironmentMesh(join(ompl_resources_dir, 'Home_env.dae'))
setup.getSpaceInformation().setStateValidityCheckingResolution(0.01)
setup.setPlanner(allocatePlanner(setup.getSpaceInformation(), 'rrtstar'))
setup.setup()
si = setup.getSpaceInformation()



def QtoAxisAngle(Q):
    # angle = 2 * acos(qw)
    #x = qx / sqrt(1-qw*qw)
    #y = qy / sqrt(1-qw*qw)
    #z = qz / sqrt(1-qw*qw)
    # to unit quarternion
    Q = Q / np.linalg.norm(Q)
    angle = 2 * np.arccos(Q[0])
    # for testing singularity
    if Q[0]*Q[0] == 1.0:
        # then can be set to any arbitrary value
        x = 1.0
        y = 0.0
        z = 0.0
    else:
        x = Q[1] / np.sqrt(1-Q[0]*Q[0])
        y = Q[2] / np.sqrt(1-Q[0]*Q[0])
        z = Q[3] / np.sqrt(1-Q[0]*Q[0])

    return np.array([x, y, z, angle])

def removeCollision(path, obc, IsInCollision):
    new_path = []
    # rule out nodes that are already in collision
    for i in range(0,len(path)):
        if not IsInCollision(path[i].numpy(),obc):
            #print('point not in collision')
            new_path.append(path[i])
        else:
            pass
            #print('point in collision')
    return new_path

def steerTo(start, end, obc, IsInCollision, step_sz=DEFAULT_STEP):
    # test if there is a collision free path from start to end, with step size
    # given by step_sz, and with generic collision check function
    # here we assume start and end are tensors
    # return 0 if in coliision; 1 otherwise
    start_t = time.time()

    start_ompl = ob.State(setup.getSpaceInformation())
    start_ompl().setX(start[0].item())
    start_ompl().setY(start[1].item())
    start_ompl().setZ(start[2].item())
    angle = np.array([start[6].item(), start[3].item(), start[4].item(), start[5].item()])
    angle = QtoAxisAngle(angle)
    start_ompl().rotation().setAxisAngle(angle[0], angle[1], angle[2], angle[3])

    end_ompl = ob.State(setup.getSpaceInformation())
    end_ompl().setX(end[0].item())
    end_ompl().setY(end[1].item())
    end_ompl().setZ(end[2].item())
    angle = np.array([end[6].item(), end[3].item(), end[4].item(), end[5].item()])
    angle = QtoAxisAngle(angle)
    end_ompl().rotation().setAxisAngle(angle[0], angle[1], angle[2], angle[3])

    path_ompl = og.PathGeometric(si)
    path_ompl.append(start_ompl())
    path_ompl.append(end_ompl())
    return path_ompl.check()


def feasibility_check(path, obc, IsInCollision, step_sz=DEFAULT_STEP):
    # checks the feasibility of entire path including the path edges
    # by checking for each adjacent vertices
    for i in range(0,len(path)-1):
        if not steerTo(path[i],path[i+1],obc,IsInCollision,step_sz=step_sz):
            # collision occurs from adjacent vertices
            return 0
    return 1


def dist_lvc(path, obc, IsInCollision, step_sz=DEFAULT_STEP):
    """
    this function first reorder the path by distance,
    then use lvc to smooth the path
    detail:
        1) reorder the path except the goal node, then append the goal node at the end
        |
        v
        2) also check if goal is the possible next state, if so, ignore other nodes
    """

    # reorder by distance
    #- simple algorithm: linear search all remaining nodes
    #new_path = [path[0]]
    new_path_idx = [0]
    prev_idx = 0
    for i in range(len(path)-1):
        # obtain the nearest neighbor in the path not picked
        min_dist = 1e8
        min_j = -1
        for j in range(1, len(path)):
            # we ignore the goal node so that we can make sure the goal is always the last node
            if j in new_path_idx:
                continue
            # calculate the distance
            dist = torch.norm(path[prev_idx]-path[j])
            if dist < min_dist:
                min_dist = dist
                min_j = j
        new_path_idx.append(min_j)
        # now we are finding path from min_j
        prev_idx = min_j
        if min_j == len(path)-1:
            # goal node
            break
    new_path = [path[i] for i in new_path_idx]
    return new_path
    #return lvc(new_path, obc, IsInCollision, step_sz)


def lvc(path, obc, IsInCollision, step_sz=DEFAULT_STEP):
    # lazy vertex contraction
    for i in range(0,len(path)-1):
        for j in range(len(path)-1,i+1,-1):
            ind=0
            ind=steerTo(path[i],path[j],obc,IsInCollision,step_sz=step_sz)
            if ind==1:
                pc=[]
                for k in range(0,i+1):
                    pc.append(path[k])
                for k in range(j,len(path)):
                    pc.append(path[k])
                return lvc(pc,obc,IsInCollision,step_sz=step_sz)
    return path

def neural_replan(mpNet, path, obc, obs, IsInCollision, normalize, unnormalize, init_plan_flag, step_sz=DEFAULT_STEP, time_flag=False):
    if init_plan_flag:
        # if it is the initial plan, then we just do neural_replan
        #MAX_LENGTH = 80
        MAX_LENGTH = 2400
        mini_path, time_d = neural_replanner(mpNet, path[0], path[-1], obc, obs, IsInCollision, \
                                            normalize, unnormalize, MAX_LENGTH, step_sz=step_sz)
        if mini_path:
            if time_flag:
                return removeCollision(mini_path, obc, IsInCollision), time_d
                #return mini_path, time_d
            else:
                return removeCollision(mini_path, obc, IsInCollision)
                #return mini_path
        else:
            # can't find a path
            if time_flag:
                return path, time_d
            else:
                return path
    #MAX_LENGTH = 50
    MAX_LENGTH = 3000
    # replan segments of paths
    new_path = [path[0]]
    time_norm = 0.
    for i in range(len(path)-1):
        # look at if adjacent nodes can be connected
        # assume start is already in new path
        start = path[i]
        goal = path[i+1]

        steer = steerTo(start, goal, obc, IsInCollision, step_sz=step_sz)
        if steer:
            new_path.append(goal)
        else:
            # plan mini path
            mini_path, time_d = neural_replanner(mpNet, start, goal, obc, obs, IsInCollision, \
                                                normalize, unnormalize, MAX_LENGTH, step_sz=step_sz)
            time_norm += time_d
            if mini_path:
                path_to_add = removeCollision(mini_path[1:], obc, IsInCollision)
                # edit: NN reorder for local plan
                # edit: may also add lvc
                path_to_add = dist_lvc(path_to_add, obc, IsInCollision, step_sz)

                new_path += path_to_add
                #new_path += removeCollision(mini_path[1:], obc, IsInCollision)  # take out start point
                #new_path += mini_path[1:]
            else:
                #new_path += path[i+1:]     # just take in the rest of the path
                #break
                #edit: we can still plan the rest of the path, even if from path[i] -> path[i+1] fail
                new_path.append(goal)
    if time_flag:
        return new_path, time_norm
    else:
        return new_path


def neural_replanner(mpNet, start, goal, obc, obs, IsInCollision, normalize, unnormalize, MAX_LENGTH, step_sz=DEFAULT_STEP):
    # plan a mini path from start to goal
    # obs: tensor
    itr=0
    pA=[]
    pA.append(start)
    pB=[]
    pB.append(goal)
    target_reached=0
    tree=0
    #tree=1  # turn this off for bidirectional
    new_path = []
    time_norm = 0.
    #print('neural replan:')
    #print('start:')
    #print(start)
    #print('goal:')
    #print(goal)
    vis_start = start
    vis_goal = goal
    while target_reached==0 and itr<MAX_LENGTH:
        itr=itr+1  # prevent the path from being too long
        if tree==0:
            ip1 = torch.cat((start, goal)).unsqueeze(0)
            ob1 = torch.FloatTensor(obs).unsqueeze(0)
            #ip1=torch.cat((obs,start,goal)).unsqueeze(0)
            time0 = time.time()
            ip1=normalize(ip1)
            time_norm += time.time() - time0
            ip1=to_var(ip1)
            ob1=to_var(ob1)
            sample=mpNet(ip1,ob1).squeeze(0)
            # unnormalize to world size
            sample=sample.data.cpu()
            time0 = time.time()
            sample = unnormalize(sample)
            time_norm += time.time() - time0
            if not IsInCollision(sample, obc):
                start = sample
                pA.append(start)
            tree=1
            #tree=0  # turn this off to use bidirectional
        else:
            ip2 = torch.cat((goal, start)).unsqueeze(0)
            ob2 = torch.FloatTensor(obs).unsqueeze(0)
            #ip2=torch.cat((obs,goal,start)).unsqueeze(0)
            time0 = time.time()
            ip2=normalize(ip2)
            time_norm += time.time() - time0
            ip2=to_var(ip2)
            ob2=to_var(ob2)
            sample=mpNet(ip2,ob2).squeeze(0)
            # unnormalize to world size
            sample=sample.data.cpu()
            time0 = time.time()
            sample = unnormalize(sample)
            time_norm += time.time() - time0
            if not IsInCollision(sample, obc):
                goal = sample
                pB.append(goal)
            tree=0
            #tree=1  # turn this off for bidirectional
        target_reached=steerTo(start, goal, obc, IsInCollision, step_sz=step_sz)

    vis_path_pA = [p.numpy() for p in pA]
    vis_path_pA = np.array(vis_path_pA)
    vis_path_pB = [p.numpy() for p in pB]
    vis_path_pB = np.array(vis_path_pB)

    #np.savetxt('path_%f_%f_%f_to_%f_%f_%f_pA.txt' % (vis_start[0].item(),vis_start[1].item(),vis_start[2].item(),
    #                                              vis_goal[0].item(),vis_goal[1].item(),vis_goal[2].item()), vis_path_pA, fmt='%f')
    #np.savetxt('path_%f_%f_%f_to_%f_%f_%f_pB.txt' % (vis_start[0].item(),vis_start[1].item(),vis_start[2].item(),
    #                                              vis_goal[0].item(),vis_goal[1].item(),vis_goal[2].item()), vis_path_pB, fmt='%f')


    if target_reached==0:
        return 0, time_norm
    else:
        for p1 in range(len(pA)):
            new_path.append(pA[p1])
        for p2 in range(len(pB)-1,-1,-1):
            new_path.append(pB[p2])

    return new_path, time_norm


def complete_replan_global(mpNet, path, true_path, true_path_length, obc, obs, obs_i, \
                           normalize, step_sz=DEFAULT_STEP):
    # use the training dataset as demonstration (which was trained by rrt*)
    # input path: list of tensor
    # obs: tensor
    demo_path = true_path[:true_path_length]
    dataset, targets, env_indices = transformToTrain(demo_path, len(demo_path), obs, obs_i)
    added_data = list(zip(dataset,targets,env_indices))
    bi = np.array(dataset).astype(np.float32)
    bobs = obs.numpy().reshape(1,-1).repeat(len(dataset),axis=0).astype(np.float32)
    bi = torch.FloatTensor(bi)
    bobs = torch.FloatTensor(bobs)
    bt = torch.FloatTensor(targets)
    # normalize first
    bi, bt = normalize(bi), normalize(bt)
    mpNet.zero_grad()
    bi=to_var(bi)
    bobs=to_var(bobs)
    bt=to_var(bt)
    mpNet.observe(0, bi, bobs, bt)
    demo_path = [torch.from_numpy(p).type(torch.FloatTensor) for p in demo_path]
    return demo_path, added_data


def transformToTrain(path, path_length, obs, obs_i):
    dataset=[]
    targets=[]
    env_indices = []
    for m in range(0, path_length-1):
        data = np.concatenate( (path[m], path[path_length-1]) ).astype(np.float32)
        targets.append(path[m+1])
        dataset.append(data)
        env_indices.append(obs_i)
    return dataset,targets,env_indices
