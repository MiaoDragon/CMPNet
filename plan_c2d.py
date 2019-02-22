import numpy as np
def IsInCollision(x,obc):
    s=np.zeros(2,dtype=np.float32)
    #padding=0.5
    padding=0.0
    shape=[[10.0,5.0],[5.0,10.0],[10.0,10.0],[10.0,5.0],[5.0,10.0],[10.0,5.0],[5.0,10.0]]
    s[0]=x[0]
    s[1]=x[1]
    for i in range(0,7):
        cf=True
        for j in range(0,2):
            if abs(obc[idx][i][j] - s[j]) > (shape[i][j]-padding)/2.0 and s[j] < 20.0 and s[j] > -20.0:
                cf=False
                break

        if cf==True:
            return True
    return False
