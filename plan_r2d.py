import numpy as np
import math
def line_intersect(a0, a1, b0, b1):
    x1 = a1[1] - a0[1]
    y1 = a0[0] - a1[0]
    c1 = a1[0] * a0[1] - a0[0] * a1[1]
    r3 = x1 * b0[0] + y1 * b0[1] + c1
    r4 = x1 * b1[0] + y1 * b1[1] + c1
    if r3 * r4 > 0:
        return False
    x2 = b1[1] - b0[1]
    y2 = b0[0] - b1[0]
    c2 = b1[0] * b0[1] - b0[0] * b1[1]
    r1 = x2 * a0[0] + y2 * a0[1] + c2
    r2 = x2 * a1[0] + y2 * a1[1] + c2
    if r1 * r2 > 0:
        return False
    denom = x1 * y2 - x2 * y1
    if denom == 0.:
        return False  # collinear
    return True
    
def IsInCollision(stateIn,obc):
    # if origin is out of world, return True
    if abs(stateIn[0]) > 20. or abs(stateIn[1]) > 20.:
        return True
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

    for i in range(0,7):
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
