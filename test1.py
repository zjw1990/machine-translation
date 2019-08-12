# Code in file tensor/two_layer_net_numpy.py
import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w2 = np.random.randn(D_in, H)
w3 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
  # Forward pass: compute predicted y
  z = x.dot(w2)
  a = np.maximum(z,0)
  # y: 64*10
  # no activate function
  y_pred = a.dot(w3)
  # Compute and print loss
  loss = np.square(y_pred - y).sum()
  print(t, loss)
  
  sigma_3 = 2*(y_pred-y)
  w_3_grad = a.T.dot(sigma_3)
  
  sigma_2 = w3.dot(sigma_3.T)
  sigma_2[z.T<0] = 0

  w_2_grad = x.T.dot(sigma_2.T)

  # Update weights
  w2 -= learning_rate * w_2_grad
  w3 -= learning_rate * w_3_grad



