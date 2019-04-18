import Model.AE.CAE_baxter as CAE_baxter
from utility import *
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from og_data_loader import load_dataset, load_dataset_single_paths_only, load_test_dataset_end2end#load_test_dataset_single_paths_only
# from model import MLP, MLP_Path
from torch.autograd import Variable
import math
from mpnet_lib.import_tool import fileImport
import baxter_interface
import time
import rospy
import sys

sys.path.append('/home/anthony/catkin_workspaces/baxter_ws/src/baxter_mpnet/scripts')
from motion_planning_dataset import *
from motion_planning_dataset import set_environment as setup_env

from Model.GEM_end2end_model import End2EndMPNet
import Model.model_baxter as model_baxter


CAE = CAE_baxter
# MLP = model_baxter.MLP
MLP = model_baxter.MLP_Path


joint_ranges = np.array([3.4033, 3.194, 6.117, 3.6647, 6.117, 6.1083, 2.67])

params = {}
params['name'] = 'test_og_mpnet_'
params['good_name'] = 'good_path_samples/'
params['bad_name'] = 'bad_path_samples/'
params['step_size'] = 0.01
params['mlp_model'] = 'mlp_PReLU_ae_dd140'
params['cae_model'] = 'cae_encoder_140'
params['DEFAULT_STEP'] = 0.05


parser = argparse.ArgumentParser()
# for training
parser.add_argument('--model_path', type=str,
                    default='./models/', help='path for saving trained models')
parser.add_argument('--seen_N', type=int, default=0)
parser.add_argument('--seen_NP', type=int, default=0)
parser.add_argument('--seen_s', type=int, default=0)
parser.add_argument('--seen_sp', type=int, default=0)
parser.add_argument('--unseen_N', type=int, default=0)
parser.add_argument('--unseen_NP', type=int, default=0)
parser.add_argument('--unseen_s', type=int, default=0)
parser.add_argument('--unseen_sp', type=int, default=0)
parser.add_argument('--grad_step', type=int, default=1,
                    help='number of gradient steps in continual learning')
# for continual learning
parser.add_argument('--n_tasks', type=int, default=1, help='number of tasks')
parser.add_argument('--n_memories', type=int, default=256,
                    help='number of memories for each task')
parser.add_argument('--memory_strength', type=float, default=0.5,
                    help='memory strength (meaning depends on memory)')
# Model parameters
parser.add_argument('--total_input_size', type=int,
                    default=2800+4, help='dimension of total input')
parser.add_argument('--AE_input_size', type=int,
                    default=2800, help='dimension of input to AE')
parser.add_argument('--mlp_input_size', type=int, default=28 +
                    4, help='dimension of the input vector')
parser.add_argument('--output_size', type=int, default=2,
                    help='dimension of the input vector')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--device', type=int, default=0, help='cuda device')
parser.add_argument('--data_path', type=str, default='../data/simple/')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--memory_type', type=str, default='res',
                    help='res for reservoid, rand for random sampling')
parser.add_argument('--env_type', type=str, default='s2d',
                    help='s2d for simple 2d, c2d for complex 2d')
parser.add_argument('--world_size', type=int,
                    default=50, help='boundary of world')

parser.add_argument('--dl1', type=bool, default=True)
parser.add_argument('--docker', type=bool, default=False)

args = parser.parse_args()
print(args)

torch_seed = np.random.randint(low=0, high=1000)
np_seed = np.random.randint(low=0, high=1000)
py_seed = np.random.randint(low=0, high=1000)
torch.manual_seed(torch_seed)
np.random.seed(np_seed)
random.seed(py_seed)

# Build the models
if torch.cuda.is_available():
    torch.cuda.set_device(args.device)

# DEFAULT_STEP = params['DEFAULT_STEP']
DEFAULT_STEP = 0.05

# mpNet = End2EndMPNet(args.total_input_size, args.AE_input_size, args.mlp_input_size,
#                      args.output_size, 'deep', args.n_tasks, args.n_memories, args.memory_strength, args.grad_step,
#                      CAE, MLP)

mpNet = End2EndMPNet(total_input_size=16067, AE_input_size=16053, mlp_input_size=74, output_size=7, AEtype='deep', \
            n_tasks=1, n_memories=10000, memory_strength=0.5, grad_step=1, CAE=CAE, MLP=MLP, train=False)

print("CAE:")
print(mpNet.encoder)
print("MLP:")
print(mpNet.mlp)

# load previously trained model if start epoch > 0
model_path = 'cmpnet_epoch_%d.pkl' % (args.start_epoch)
if args.start_epoch > 0:
    load_net_state(mpNet, os.path.join(args.model_path, model_path))
    torch_seed, np_seed, py_seed = load_seed(
        os.path.join(args.model_path, model_path))
    # set seed after loading
    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)
    random.seed(py_seed)

if torch.cuda.is_available():
    mpNet.cuda()
    mpNet.mlp.cuda()
    mpNet.encoder.cuda()
    mpNet.set_opt(torch.optim.Adagrad, lr=args.learning_rate)

if args.start_epoch > 0:
    load_opt_state(mpNet, os.path.join(args.model_path, model_path))

# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(nn.Linear(16053, 786), nn.PReLU(),
#                                      nn.Linear(786, 512), nn.PReLU(),
#                                      nn.Linear(512, 256), nn.PReLU(),
#                                      nn.Linear(256, 60))

#     def forward(self, x):
#         x = self.encoder(x)
#         return x


# class Encoder_End2End(nn.Module):
#     def __init__(self):
#         super(Encoder_End2End, self).__init__()
#         self.encoder = nn.Sequential(nn.Linear(16053, 256), nn.PReLU(),
#                                      nn.Linear(256, 256), nn.PReLU(),
#                                      nn.Linear(256, 60))

#     def forward(self, x):
#         x = self.encoder(x)
#         return x

def IsInCollision(state, print_depth=False):

    # returns true if robot state is in collision, false if robot state is collision free
    filler_robot_state[10:17] = moveit_scrambler(np.multiply(state,joint_ranges))
    rs_man.joint_state.position = tuple(filler_robot_state)

    col_start = time.clock()
    collision_free = sv.getStateValidity(rs_man, group_name="right_arm", print_depth=print_depth)
    col_end = time.clock()
    col_time = col_end - col_start

    global counter
    global col_time_env

    counter += 1
    col_time_env.append(col_time)
    return (not collision_free)


def steerTo (start, end, step_sz=DEFAULT_STEP, print_depth=False, dof=7):

    DISCRETIZATION_STEP=step_sz
    dists=np.zeros(dof,dtype=np.float32)
    for i in range(0,dof):
        dists[i] = end[i] - start[i]

    distTotal = 0.0
    for i in range(0,dof):
        distTotal =distTotal+ dists[i]*dists[i]

    distTotal = math.sqrt(distTotal)
    if distTotal>0:
        incrementTotal = distTotal/DISCRETIZATION_STEP
        for i in range(0,dof):
            dists[i] =dists[i]/incrementTotal

        numSegments = int(math.floor(incrementTotal))

        stateCurr = np.zeros(7,dtype=np.float32)
        for i in range(0,dof):
            stateCurr[i] = start[i]
        for i in range(0,numSegments):

            if IsInCollision(stateCurr, print_depth=print_depth):
                return 0

            for j in range(0,dof):
                stateCurr[j] = stateCurr[j]+dists[j]


        if IsInCollision(end, print_depth=print_depth):
            return 0


    return 1

# checks the feasibility of entire path including the path edges
def feasibility_check(path, step_sz=DEFAULT_STEP, print_depth=False):

    for i in range(0,len(path)-1):
        ind=steerTo(path[i],path[i+1], step_sz=step_sz, print_depth=print_depth)
        if ind==0:
            return 0
    return 1


# checks the feasibility of path nodes only
def collision_check(path):

    for i in range(0,len(path)):
        if IsInCollision(path[i]):
            return 0
    return 1

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def get_input(i,dataset,targets,seq,bs):
    bi=np.zeros((bs,18),dtype=np.float32)
    bt=np.zeros((bs,2),dtype=np.float32)
    k=0
    for b in range(i,i+bs):
        bi[k]=dataset[seq[i]].flatten()
        bt[k]=targets[seq[i]].flatten()
        k=k+1
    return torch.from_numpy(bi),torch.from_numpy(bt)



def is_reaching_target(start1,start2,dof=7):

    s1=np.zeros(dof, dtype=np.float32)
    for i in range(dof):
        s1[i] = start1[i]

    s2=np.zeros(dof, dtype=np.float32)
    for i in range(dof):
        s2[i] = start2[i]

    for i in range(0,dof):
        if abs(s1[i]-s2[i]) > 0.05:
            return False
    return True

#lazy vertex contraction
def lvc(path, step_sz=DEFAULT_STEP):

    for i in range(0,len(path)-1):
        for j in range(len(path)-1,i+1,-1):
            ind=0
            ind=steerTo(path[i],path[j],step_sz=step_sz)
            if ind==1:
                pc=[]
                for k in range(0,i+1):
                    pc.append(path[k])
                for k in range(j,len(path)):
                    pc.append(path[k])

                return lvc(pc)

    return path

def re_iterate_path2(p,g):
    step=0
    path=[]
    path.append(p[0])
    for i in range(1,len(p)-1):
        if not IsInCollision(p[i]):
            path.append(p[i])
    path.append(g)
    new_path=[]
    for i in range(0,len(path)-1):
        target_reached=False


        st=path[i]
        gl=path[i+1]
        steer=steerTo(st, gl)
        if steer==1:
            new_path.append(st)
            new_path.append(gl)
        else:
            itr=0
            target_reached=False
            while (not target_reached) and itr<50 :
                new_path.append(st)
                itr=itr+1
                ip=torch.cat((st,gl))
                ip=to_var(ip)
                st=mlp(ip)
                st=st.data.cpu()
                target_reached=is_reaching_target(st,gl)
            if target_reached==False:
                return 0

    #new_path.append(g)
    return new_path

def replan_path(p,g,obs=None,step_sz=DEFAULT_STEP):
    if obs is None:

        step=0
        path=[]
        path.append(p[0])
        for i in range(1,len(p)-1):
            if not IsInCollision(p[i]):
                path.append(p[i])
        path.append(g)
        new_path=[]
        for i in range(0,len(path)-1):
            target_reached=False


            st=path[i]
            gl=path[i+1]
            steer=steerTo(st, gl)
            if steer==1:
                new_path.append(st)
                new_path.append(gl)
            else:
                itr=0
                pA=[]
                pA.append(st)
                pB=[]
                pB.append(gl)
                target_reached=0
                tree=0
                while target_reached==0 and itr<3000 :
                    itr=itr+1
                    if tree==0:
                        ip1=torch.cat((st,gl))
                        ip1=to_var(ip1)
                        st=mpNet(ip1)
                        st=st.data.cpu()
                        pA.append(st)
                        tree=1
                    else:
                        ip2=torch.cat((gl,st))
                        ip2=to_var(ip2)
                        gl=mpNet(ip2)
                        gl=gl.data.cpu()
                        pB.append(gl)
                        tree=0
                    target_reached=steerTo(st, gl)
                if target_reached==0:
                    print("failed to replan")
                    return 0
                else:
                    for p1 in range(0,len(pA)):
                        new_path.append(pA[p1])
                    for p2 in range(len(pB)-1,-1,-1):
                        new_path.append(pB[p2])

        return new_path

    else:
        step=0
        path=[]
        path.append(p[0])
        for i in range(1,len(p)-1):
            if not IsInCollision(p[i]):
                path.append(p[i])
        path.append(g)
        new_path=[]
        for i in range(0,len(path)-1):
            target_reached=False


            st=path[i]
            gl=path[i+1]
            steer=steerTo(st, gl, step_sz=step_sz)
            if steer==1:
                new_path.append(st)
                new_path.append(gl)
            else:
                itr=0
                pA=[]
                pA.append(st)
                pB=[]
                pB.append(gl)
                target_reached=0
                tree=0
                while target_reached==0 and itr<3000 :
                    itr=itr+1
                    if tree==0:
                        # ip1=torch.cat((st,gl,obs.data.cpu()))
                        ip1=torch.cat((obs.data.cpu(), st, gl)) #flipped for cmpnet training
                        ip1=to_var(ip1)
                        # st=mlp(ip1)
                        st = mpNet(ip1)
                        st=st.data.cpu()
                        pA.append(st)
                        tree=1
                    else:
                        # ip2=torch.cat((gl,st,obs.data.cpu()))
                        ip2=torch.cat((obs.data.cpu(), gl, st))                        
                        ip2=to_var(ip2)
                        # gl=mlp(ip2)
                        gl = mpNet(ip2)
                        gl=gl.data.cpu()
                        pB.append(gl)
                        tree=0
                    target_reached=steerTo(st, gl, step_sz=step_sz)
                if target_reached==0:
                    print("failed to replan")
                    return 0
                else:
                    for p1 in range(0,len(pA)):
                        new_path.append(pA[p1])
                    for p2 in range(len(pB)-1,-1,-1):
                        new_path.append(pB[p2])

        return new_path



def save_feasible_path(path, filename):
    with open(filename+'.pkl', 'wb') as good_f:
        pickle.dump(path, good_f)



def check_full_path(overall_path):
    invalid = []

    valid = True
    overall_valid = True
    for i, state in enumerate(overall_path):
        filler_robot_state[10:17] = moveit_scrambler(state)
        rs_man.joint_state.position = tuple(filler_robot_state)
        collision_free = sv.getStateValidity(rs_man, group_name="right_arm")

        valid = valid and collision_free
        overall_valid = overall_valid and collision_free

        if not valid:
            invalid.append(i)
            valid = True

    if (len(invalid)==0 and overall_valid):
        print("Full path valid!")
    else:
        print("Not valid")

    return overall_valid

def path_to_np(path):
    path_np = []
    for i, state in enumerate(path):
        path_np.append(np.multiply(state.numpy(), joint_ranges)) #unnormalize
    return path_np

def make_overall_path(path_np):
    dists = []
    for i in range(1,len(path_np)):
        dists.append(np.mean(abs(path_np[i] - path_np[i-1]), axis=0))

    overall_dist = sum(dists)
    fractions = [x for x in dists/overall_dist]
    total_pts = 300
    pts = [int(total_pts * x) for x in fractions]
    path_full = []
    for i, n_pts in enumerate(pts):
        vec = np.transpose(np.linspace(path_np[i][0], path_np[i+1][0], n_pts)[np.newaxis])
        for j in range(1,7):
            vec = np.hstack([vec, np.transpose(np.linspace(path_np[i][j], path_np[i+1][j], n_pts)[np.newaxis])])
        path_full.append(vec)

    overall_path = []
    for mini_path in path_full:
        for state in mini_path:
            overall_path.append(state)

    return overall_path

def play_smooth(overall_path, limb):

    done = False
    joint_state= limb.joint_angles()

    while not done:
        for i, name in enumerate(joint_state.keys()):
            joint_state[name] = overall_path[k][i]
        limb.set_joint_positions(joint_state)
        time.sleep(0.025)
        k += 1
        if k > len(overall_path)-1:
            done = True


importer = fileImport()
env_data_path = '/home/anthony/catkin_workspaces/baxter_ws/src/baxter_mpnet/data/full_dataset_sample/'
# env_data_path = '/baxter_mpnet_docker/data/full_dataset_sample/'
pcd_data_path = env_data_path+'pcd/'
envs_file = 'trainEnvironments_GazeboPatch.pkl'

envs = importer.environments_import(env_data_path + envs_file)

envs_load = [envs[0]]
print(envs_load)
obstacles, paths, path_lengths = load_test_dataset_end2end(envs_load, env_data_path, pcd_data_path, importer, NP=100)

### end2end version ###
# Build the models

# model_path = '/baxter_mpnet_docker/mpnet/MPNet/models_end2end_all_envs_STAR_0/'
# cae_path = '/baxter_mpnet_docker/mpnet/MPNet/models_end2end_all_envs_STAR_0/'

# mlp = MLP(60+7*2, 7)
# encoder = Encoder_End2End()

# device = torch.device('cpu')

# mlp.load_state_dict(torch.load(model_path+params['mlp_model']+'.pkl', map_location=device))
# encoder.load_state_dict(torch.load(cae_path+params['cae_model']+'.pkl', map_location=device))

# if torch.cuda.is_available():
#     encoder.cuda()
    # mlp.cuda()


with open (env_data_path+envs_file, 'rb') as env_f:
    envDict = pickle.load(env_f)

# home_path = '/baxter_mpnet_docker/'

# for i, key in enumerate(envDict['obsData'].keys()):
#     fname = envDict['obsData'][key]['mesh_file']

#     if fname is not None:
#         keep = fname.split('/baxter_mpnet/')[1]
#         new = home_path + keep
#         print(new)
#         envDict['obsData'][key]['mesh_file'] = new

rospy.init_node("environment_monitor")
# limb = baxter_interface.Limb('right')
scene = PlanningSceneInterface()
robot = RobotCommander()
group = MoveGroupCommander("right_arm")
scene._scene_pub = rospy.Publisher('planning_scene',
                                      PlanningScene,
                                      queue_size=0)

sv = StateValidity()
setup_env(robot, scene)

masterModifier = ShelfSceneModifier()
sceneModifier = PlanningSceneModifier(envDict['obsData'])
sceneModifier.setup_scene(scene, robot, group)



rs_man = RobotState()
robot_state = robot.get_current_state()
rs_man.joint_state.name = robot_state.joint_state.name
filler_robot_state = list(robot_state.joint_state.position)



dof=7

tp=0
fp=0
tot=[]
et_tot = []
neural_paths = {}
bad_paths = {}

goal_collision = []

experiment_name = params['name']
paths_path = params['good_name']+experiment_name
bad_paths_path = params['bad_name']+experiment_name

global counter
global col_time
global col_time_env

for i, env_name in enumerate(envs_load):
    et=[]
    col_env = []
    tp_env = 0
    fp_env = 0
    neural_paths[env_name] = []
    bad_paths[env_name] = []

    global col_time_env
    col_time_env = []


    print("ENVIRONMENT: " + env_name)

    sceneModifier.delete_obstacles()
    new_pose = envDict['poses'][env_name]
    sceneModifier.permute_obstacles(new_pose)

    for j in range(15, path_lengths.shape[1]):
    # for j in range(0,path_lengths.shape[1]):
        print ("step: i="+str(i)+" j="+str(j))
        print("fp: " + str(fp_env))
        print("tp: " + str(tp_env))
        p1_ind=0
        p2_ind=0
        p_ind=0

        obs=obstacles[i]
        # obs=obstacles[i][:, np.newaxis] #to get (1, 16053)
        # obs=obs.T
        obs=torch.from_numpy(obs)
        # print(obs.size())
        obs=to_var(obs)

        # en_inp=to_var(obs)
        # h=encoder(en_inp)

        if path_lengths[i][j]>0:
            global counter
            global col_time
            counter = 0

            if (j%10 == 0):
                print("running avergae collision time: ")
                print(np.mean(col_time_env))

            # print(path_lengths[i][j])
            start=np.zeros(dof, dtype=np.float32)
            goal=np.zeros(dof, dtype=np.float32)
            for l in range(0,dof):
                start[l]=paths[i][j][0][l]

            for l in range(0,dof):
                goal[l]=paths[i][j][path_lengths[i][j]-1][l]

            if (IsInCollision(goal)):
                print("GOAL IN COLLISION --- BREAKING")
                goal_collision.append(j)
                continue

            start1=torch.from_numpy(start)
            goal2=torch.from_numpy(start)
            goal1=torch.from_numpy(goal)
            start2=torch.from_numpy(goal)

            ##generated paths
            path1=[]
            path1.append(start1)
            path2=[]
            path2.append(start2)
            path=[]
            target_reached=0
            step=0
            path=[] # stores end2end path by concatenating path1 and path2
            tree=0
            tic_start = time.clock()
            step_sz = DEFAULT_STEP

            print("in while loop\n")
            while target_reached==0 and step<3000:
                step=step+1
                if (j > 15):
                    print("step: " + str(step))
                    print("start: ")
                    print(start1)
                    print("goal: ")
                    print(start2)
                if tree==0:
                    # inp1=torch.cat((obs.data.cpu(),start1,start2), 0).unsqueeze(1)
                    inp1=torch.cat((obs.data.cpu(),start1,start2))
                    # print(inp1.size())
                    # inp1=inp1.transpose(0,1)
                    # print(inp1.size())
                    inp1=to_var(inp1)
                    start1=mpNet(inp1)
                    start1=start1.data.cpu()
                    path1.append(start1)
                    tree=1
                else:
                    # print(start1.size())
                    # print(start2.size())
                    # inp2=torch.cat((obs.data.cpu(),start2,start1), 0).unsqueeze(1)
                    inp2=torch.cat((obs.data.cpu(),start2,start1))
                    # print(inp2.size())
                    # inp2=inp2.transpose(0,1)
                    # print(inp2.size())
                    inp2=to_var(inp2)
                    start2=mpNet(inp2)
                    start2=start2.data.cpu()
                    path2.append(start2)
                    tree=0
                target_reached=steerTo(start1,start2)

            tp=tp+1
            tp_env=tp_env+1

            if (step > 3000 or not target_reached):
                save_feasible_path(path, bad_paths_path+env_name+'_bp_'+str(j))

            if target_reached==1:
                for p1 in range(0,len(path1)):
                    path.append(path1[p1])
                for p2 in range(len(path2)-1,-1,-1):
                    path.append(path2[p2])

                path=lvc(path, step_sz=step_sz)

                # full dense collision check
                indicator=feasibility_check(path, step_sz=0.01, print_depth=True)

                if indicator==1:
                    toc = time.clock()

                    t=toc-tic_start
                    et.append(t)
                    col_env.append(counter)
                    fp=fp+1
                    fp_env=fp_env+1
                    neural_paths[env_name].append(path)
                    save_feasible_path(path, paths_path+env_name+'_fp_'+str(j))
                    print("---path found---")
                    print("length: " + str(len(path)))
                    print("time: " + str(t))
                    print("count: " + str(counter))
                else:
                    sp=0
                    indicator=0
                    step_sz = DEFAULT_STEP
                    while indicator==0 and sp<10 and path !=0:
                        print("in replanning, step: " + str(sp))

                        # adaptive step size on replanning attempts
                        if (sp == 1):
                            step_sz = 0.04
                        elif (sp == 2):
                            step_sz = 0.03
                        elif (sp > 2):
                            step_sz = 0.02

                        sp=sp+1
                        g=np.zeros(dof,dtype=np.float32)
                        g=torch.from_numpy(paths[i][j][path_lengths[i][j]-1])

                        tic = time.clock()
                        path=replan_path(path,g,obs,step_sz=step_sz) #replanning at coarse level
                        toc = time.clock()

                        if path !=0:
                            path=lvc(path,step_sz=step_sz)

                            # full dense collision check
                            indicator=feasibility_check(path,step_sz=0.01,print_depth=True)

                            if indicator==1:
                                toc = time.clock()

                                t=toc-tic_start
                                et.append(t)
                                col_env.append(counter)
                                fp=fp+1
                                fp_env=fp_env+1
                                neural_paths[env_name].append(path)
                                save_feasible_path(path, paths_path+env_name+'_fp_'+str(j))

                                print("---path found---")
                                print("length: " + str(len(path)))
                                print("time: " + str(t))
                                print("count: " + str(counter))

                    if (sp == 10):
                        save_feasible_path(path, bad_paths_path+env_name+'_bp_'+str(j))

    et_tot.append(et)

    print("total env paths: ")
    print(tp_env)
    print("feasible env paths: ")
    print(fp_env)
    print("average collision checks: ")
    print(np.mean(col_env))
    print("average time per collision check: ")
    print(np.mean(col_time_env))
    print("average time: ")
    print(np.mean(et))
    env_data = {}
    env_data['tp_env'] = tp_env
    env_data['fp_env'] = fp_env
    env_data['et_env'] = et
    env_data['col_env'] = col_env
    env_data['avg_col_time'] = np.mean(col_time_env)
    env_data['paths'] = neural_paths[env_name]

    with open(paths_path+env_name+'_env_data.pkl', 'wb') as data_f:
        pickle.dump(env_data, data_f)

print("total paths: ")
print(tp)
print("feasible paths: ")
print(fp)

with open(paths_path+'neural_paths.pkl', 'wb') as good_f:
    pickle.dump(neural_paths, good_f)

with open(paths_path+'elapsed_time.pkl', 'wb') as time_f:
    pickle.dump(et_tot, time_f)

print(np.mean([np.mean(x) for x in et_tot]))
print(np.std([np.mean(x) for x in et_tot]))

acc = []

for i, env in enumerate(envs_load):
    with open (paths_path+env+'_env_data.pkl', 'rb') as data_f:
        data = pickle.load(data_f)
    acc.append(100.0*data['fp_env']/data['tp_env'])
    print("env: " + env)
    print("accuracy: " + str(100.0*data['fp_env']/data['tp_env']))
    print("time: " + str(np.mean(data['et_env'])))
    print("min time: " + str(np.min(data['et_env'])))
    print("max time: " + str(np.max(data['et_env'])))
    print("\n")
        
print(np.mean(acc))
print(np.std(acc))
