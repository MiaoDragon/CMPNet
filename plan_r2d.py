import numpy as np
import math
size=4
def overlap(b1corner,b1axis,b1orign,b2corner,b2axis,b2orign):
    for a in range(0,2):
        t=b1corner[0][0]*b2axis[a][0]+b1corner[0][1]*b2axis[a][1]

        tMin = t
        tMax = t
        for c in range(1,4):
            t = b1corner[c][0]*b2axis[a][0]+b1corner[c][1]*b2axis[a][1]
            if t < tMin:
                tMin = t
            elif t > tMax:
                tMax = t
        if ((tMin > (1+ b2orign[a])) or (tMax < b2orign[a])):
            return False

    return True

def IsInCollision(stateIn,obc):
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
    #print "robot cornor"
    for i in range(0,4):
        #print "("+str(robot_corner[i][0])+","+str(robot_corner[i][1])+")"
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

        cf=overlap(robot_corner,robot_axis,robot_orign,obs_corner,obs_axis,obs_orign)
        if cf==True:
            return True
    return False
