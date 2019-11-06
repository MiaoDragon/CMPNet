import numpy as np
import math
size=4
import sys
from os.path import join
from ompl import base as ob
from ompl import app as oa
from ompl import geometric as og
ompl_app_root = "FILL IN PATH"
ompl_resources_dir = join(ompl_app_root, 'resources/3D')

setup = oa.SE3RigidBodyPlanning()
setup.setRobotMesh(join(ompl_resources_dir, 'Home_robot.dae'))
setup.setEnvironmentMesh(join(ompl_resources_dir, 'Home_env.dae'))
setup.getSpaceInformation().setStateValidityCheckingResolution(0.01)
setup.setup()
val = setup.getStateValidityChecker()

def QtoAxisAngle(Q):
    # angle = 2 * acos(qw)
    #x = qx / sqrt(1-qw*qw)
    #y = qy / sqrt(1-qw*qw)
    #z = qz / sqrt(1-qw*qw)
    angle = 2 * np.arccos(Q[0])
    x = Q[1] / np.sqrt(1-Q[0]*Q[0])
    y = Q[2] / np.sqrt(1-Q[0]*Q[0])
    z = Q[3] / np.sqrt(1-Q[0]*Q[0])
    return np.array([x, y, z, angle])



def IsInCollision(stateIn,obc):
    lower = np.array([-383.8, -371,47, -0.2, -1, -1, -1, -1])
    higher = np.array([325, 337.89, 142.33, 1, 1, 1, 1])
    for i in range(len(lower)):
        if stateIn[i] < lower[i] or stateIn[i] > higher[i]:
            return True  # outside of boundary
    state = ob.State(setup.getSpaceInformation())
    state().setX(stateIn[0].item())
    state().setY(stateIn[1].item())
    state().setZ(stateIn[2].item())
    # the output of RRT* quarternion seems to be in the order w, x, y, z
    angle = np.array([stateIn[6].item(), stateIn[3].item(), stateIn[4].item(), stateIn[5].item()])
    angle = QtoAxisAngle(angle)
    state().rotation().setAxisAngle(angle)
    return val.isValid(state())
