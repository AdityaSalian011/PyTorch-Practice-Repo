import torch

## check if cuda is avaible for pytorch
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'No GPU Device.')

# Everything in pytorch is based on Tensor operations.
# A tensor can have different dimensions
# so it can be 1d, 2d, or even 3d and higher

# scalar, vector, matrix, tensor

## 1. empty tensor
# x = torch.empty(1)
# x = torch.empty(2, 3)
# x = torch.empty(2, 3, 2)
# x = torch.empty(2, 3, 2, 4)
# print(x)

## 2. setting random values in tensor
# x = torch.rand(5, 3)  ## 5r, 3c tensor with random ahh values
# print(x)

## 3. tensor with 0's as values
# x = torch.zeros(5, 3)
# print(x)

# y = torch.ones(5, 3)
# print(y)

## check tensor size
# print(y.size())

## check dtype
# print(y.dtype)  ## by default float32

## specify dtype
# x = torch.ones(5,3, dtype=torch.float16)
# print(x.dtype)

## 4. construct a tensor
# x = torch.tensor([5.5, 3], dtype=torch.double)
# print(x)

## 5. arithmetic operations
# x = torch.rand(2, 2)
# y = torch.rand(2, 2)

# z = x + y
# z = torch.add(x,y)
# print(z)
# print(y.add_(x))

# substraction
# z = x - y
# z = torch.sub(x, y)

# multiplication
# z = x * y
# z = torch.mul(x,y)

# division
# z = x / y
# z = torch.div(x,y)

## 6. Slicing
# x = torch.rand(5, 3)
# print(x)
# print(x[:, 2])
# print(x[1, :])
# print(x[1,1])

# Get the actual value if only 1 element in your tensor
# print(x[1,1].item())

## 7. Reshape tensor 
# x = torch.rand(4,4)
# print(x)
# y = x.view(16)
# print(y)
# print(y.size())

# z = x.view(-1, 8)
# print(z)
# print(z.size())

## 8. Numpy
# Converting a Torch Tensor to a NumPy array and vice versa is very easy
import numpy as np

# a = torch.ones(5)
# print(a)

# b = a.numpy()
# print(b)
# print(type(b))

# Carful: If the Tensor is on the CPU (not the GPU),
# both objects will share the same memory location, so changing one
# will also change the other

# a.add_(1)
# print(a)
# print(b)

# numpy to torch with .from_numpy(x)
# a = np.ones(5)
# print(a)

# b = torch.from_numpy(a)
# print(b)

# again be careful when modifying
# a += 1
# print(a)
# print(b)

# by default all tensors are created on the CPU,
# but you can also move them to the GPU (only if it's available )

# requires_grad argument
# This will tell pytorch that it will need to calculate the gradients for this tensor
# later in your optimization steps
# i.e. this is a variable in your model that you want to optimize
x = torch.tensor([5.5, 3], requires_grad=True)
print(x)