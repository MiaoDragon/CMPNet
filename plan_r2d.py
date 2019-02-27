import numpy as np
import math
def in_between(p, a0, a1):
    # check whether p is in between a0 and a1
    # by solving p = a0 + alpha * (a1 - a0)
    dif, t = p - a0, a1 - a0
    # then dif = alpha * t
    # then |alpha| = ||dif|| / ||t||
    alpha = np.linalg.norm(dif) / np.linalg.norm(t)
    # alpha * t = +- dif
    d1 = sum(alpha * t + dif)
    d2 = sum(alpha * t - dif)
    if alpha >= 0. and alpha <= 1. and (d1 == 0. or d2 == 0.):
        return True
    return False

def line_intersect(a0,a1,b0,b1):
    """
    # given one line [a0,a1], one line [b0,b1], decide whether they intersect
    # l1: x = a0 + alpha * (a1-a0)  for alpha in [0,1]
    # l2: x = b0 + beta * (b1-b0)   for beta in [0,1]
    # obtaining system: a0 + alpha * (a1 - a0) = b0 + beta * (b1 - b0)
    # hence (alpha, beta) = [-(a1-a0), b1-b0]^{-1} (a0 - b0)
    # Notice that matrix [-(a1-a0), b1-b0] should be invertible
    # if it is not, then the two lines are parallel
    # then if there is one vertex in the line segment between the other
    # collide
    """
    # varaible recording difference
    t1, t2 = a1 - a0, b1 - b0
    mat = np.array([-t1, t2]).T  # np array saves as row vectors, but we want col
    # check if det is 0
    if np.linalg.det(mat) == 0:
        # parallel line, need to check whether they collide
        # by checking if each point is in the other line
        if in_between(a0,b0,b1):
            return True
        if in_between(a1,b0,b1):
            return True
        if in_between(b0,a0,a1):
            return True
        if in_between(b1,a0,a1):
            return True
        # otherwise parallel but no collision
        return False
    # otherwise, obtain [alpha, beta]
    sol = np.linalg.inv(mat) @ (a0 - b0)
    # if colllision point is in both lines
    if sol[0] >= 0 and sol[0] <= 1 and sol[1] >= 0 and sol[1] <= 1:
        return True
    return False

def IsInCollision(stateIn,obc):

    size = 5.
    robot_corner=np.zeros((4,2),dtype=np.float32)
    robot_axis=np.zeros((2,2),dtype=np.float32)
    robot_orign=np.zeros(2,dtype=np.float32)
    length=np.zeros(2,dtype=np.float32)
    X1=np.zeros(2,dtype=np.float32)
    Y1=np.zeros(2,dtype=np.float32)

    X1[0]=math.cos(stateIn[2])*(2.0/2.0)
    X1[1]=-math.sin(stateIn[2])*(2.0/2.0)
    Y1[0]=math.sin(stateIn[2])*(5.0/2.0)
    Y1[1]=math.cos(stateIn[2])*(5.0/2.0)

    for j in range(0,2):
        robot_corner[0][j]=stateIn[j]-X1[j]-Y1[j]
        robot_corner[1][j]=stateIn[j]+X1[j]-Y1[j]
        robot_corner[2][j]=stateIn[j]+X1[j]+Y1[j]
        robot_corner[3][j]=stateIn[j]-X1[j]+Y1[j]

        robot_axis[0][j] = robot_corner[1][j] - robot_corner[0][j]
        robot_axis[1][j] = robot_corner[3][j] - robot_corner[0][j]

    length[0]=robot_axis[0][0]*robot_axis[0][0]+robot_axis[0][1]*robot_axis[0][1]
    length[1]=robot_axis[1][0]*robot_axis[1][0]+robot_axis[1][1]*robot_axis[1][1]
    for i in range(0,4):
        pass
    for i in range(0,2):
        for j in range(0,2):
            robot_axis[i][j]=robot_axis[i][j]/float(length[j])

    robot_orign[0]=robot_corner[0][0]*robot_axis[0][0]+ robot_corner[0][1]*robot_axis[0][1]
    robot_orign[1]=robot_corner[0][0]*robot_axis[1][0]+ robot_corner[0][1]*robot_axis[1][1]

    for i in range(0,1):
        cf=True

        obs_corner=np.zeros((4,2),dtype=np.float32)
        obs_axis=np.zeros((2,2),dtype=np.float32)
        obs_orign=np.zeros(2,dtype=np.float32)
        X=np.zeros(2,dtype=np.float32)
        Y=np.zeros(2,dtype=np.float32)
        length2=np.zeros(2,dtype=np.float32)

        X[0]=1.0*size/2.0
        X[1]=0.0
        Y[0]=0.0
        Y[1]=1.0*size/2.0

        for j in range(0,2):
            obs_corner[0][j]=obc[i][j]-X[j]-Y[j]
            obs_corner[1][j]=obc[i][j]+X[j]-Y[j]
            obs_corner[2][j]=obc[i][j]+X[j]+Y[j]
            obs_corner[3][j]=obc[i][j]-X[j]+Y[j]

            obs_axis[0][j] = obs_corner[1][j] - obs_corner[0][j]
            obs_axis[1][j] = obs_corner[3][j] - obs_corner[0][j]

        length2[0]=obs_axis[0][0]*obs_axis[0][0]+obs_axis[0][1]*obs_axis[0][1]
        length2[1]=obs_axis[1][0]*obs_axis[1][0]+obs_axis[1][1]*obs_axis[1][1]

        for i1 in range(0,2):
            for j1 in range(0,2):
                obs_axis[i1][j1]=obs_axis[i1][j1]/float(length2[j1])


        obs_orign[0]=obs_corner[0][0]*obs_axis[0][0]+ obs_corner[0][1]*obs_axis[0][1]
        obs_orign[1]=obs_corner[0][0]*obs_axis[1][0]+ obs_corner[0][1]*obs_axis[1][1]
        for i in range(-1,len(robot_corner)-1):
            for j in range(-1,len(obs_corner)-1):
                if line_intersect(robot_corner[i], robot_corner[i+1], obs_corner[j], obs_corner[j+1]):
                    # if any two lines intersect, then collision
                    return True
    return False
