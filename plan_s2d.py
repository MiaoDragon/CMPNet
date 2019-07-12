import numpy as np

def IsInCollision(x,obc):
    size = 5.0
    s=np.zeros(2,dtype=np.float32)
    s[0]=x[0]
    s[1]=x[1]
    for i in range(0,7):
        cf=True
        for j in range(0,2):
            if abs(obc[i][j] - s[j]) > size/2.0 and s[j]<20.0 and s[j]>-20:
                cf=False
                break
        if cf==True:
            return True
    return False
