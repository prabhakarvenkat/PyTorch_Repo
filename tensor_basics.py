import torch

x = torch.empty(3) # 1 dimensions
#print(x)

y = torch.empty(2, 3) # 2 dimensions
#print(y)

z = torch.rand(2,3,3,2) #4 dimensions with random
#print(z)

#To view data type
#print(z.dtype)

#To view the size
#print(z.size())

#tensor from data
a = torch.tensor([2.5, 0.1])
#print(a)

'''
# Simple arithmetic operation 
r1 = torch.rand(2,2)
r2 = torch.rand(2,2)
print(r1)
print(r2)
#sum = r1 + r2
sum = torch.add(r1,r2)
print(sum)
'''

#Slicing operation
slice = torch.rand(5,3)
#print(slice[:, 0])

'''
#Reshape
re = torch.rand(4,4)
print(re)
reshape = re.view(-1, 8)
print(reshape)
'''

import numpy as np

'''
a1 = torch.ones(5)
print(a1)
b1 = a1.numpy()
print(b1)

a1.add_(1)
print(a1)
print(b1)
'''
a1 = np.ones(5)
print(a1)
b1 = torch.from_numpy(a1)
print(b1)

a1 += 1
print(a1)