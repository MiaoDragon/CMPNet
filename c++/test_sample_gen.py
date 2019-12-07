import numpy as np
# generate random test sample for comparing cpp and py
data = np.random.uniform(low=[-1.0]*14, high=[1.0]*14)
np.savetxt('test_sample.txt', data, delimiter='\n', fmt='%f')
