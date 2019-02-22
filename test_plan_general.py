from plan_general import *
def IsInCollision(x,obc):
    size = 5.0
    s=np.zeros(2,dtype=np.float32)
    s[0]=x[0]
    s[1]=x[1]
    cf = True
    for j in range(0,2):
        if abs(obc[j] - s[j]) > size/2.0:
            # not in collision
            cf=False
            break
    return cf


# test steerTo
print('steerTo test...')
start = torch.tensor([0.,-5.])
end = torch.tensor([4.99,0.])
obc = np.array([0.,0.])
print(steerTo(start, end, obc, IsInCollision, step_sz=0.01))
# test feasibility check
print('feasibility test...')
path = [torch.tensor([0.,6.]), torch.tensor([0.,2.])]
print(feasibility_check(path, obc, IsInCollision, step_sz=0.01))
# test lvc
print('lvc...')
path = [torch.tensor([0.,6.]), torch.tensor([0.,5.]),torch.tensor([0.,4.]),torch.tensor([0.,3.]),torch.tensor([-3.,3.]),torch.tensor([-3.,-3.]),
        torch.tensor([0.,-6.])]
print(lvc(path, obc, IsInCollision, step_sz=0.01))

# test neural replan
