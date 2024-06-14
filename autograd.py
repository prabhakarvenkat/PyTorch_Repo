import torch

x = torch.rand(3, requires_grad=True) #Gradient and must be true
print(x)

y = x + 2
#print(y)
z = y * y * 2
#print(z)
z = z.mean()
#print(z)

'''
z.backward() # dz/dx
print(x.grad)
'''

# Vector Jacobian 
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward() # dz/dx
print(x.grad)

# x.requires_grad_(False)
# x.detach()
# with torch.no_grad()

x.requires_grad_(False)
print(x)

y = x.detach()
print(y)

with torch.no_grad():
    y = x + 2
    print(y)

#Dummy Operation
weights = torch.ones(4, requires_grad=True)

for epoch in range(1):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)

    weights.grad.zero_() #Before performing next operation, we must empty our gradient

