'''
Author: Matt Shaffer
Contact: matt@discovermatt.com
'''


import numpy as np
import matplotlib.pyplot as plt

'''
1. Build simulated data graph
'''
steps = 1001
period = 1000
amplitude = 2
timeline = np.linspace(0, period, steps, dtype=int) # steps
num_prev = np.random.rand(1)*10
num = np.random.rand(1)*10

def rand_walk(indx):
    global num_prev
    global num
    sign = np.sign(num-num_prev)
    r = np.random.rand(1)
    j = sign*r*0.2 # value is probability that next value will change direction [default = 0.2]
    if r < 0.2:
        j = -j + num # Change direction
    else:
        j = j + num # Continue building graph in same direction
    if (num <= 0.05):
        j = np.abs(j)
    num_prev = num
    num = j
    return j

series = np.array([rand_walk(i) for i in range(steps)]) # create array for series

'''
2. Set hyperparameters, format training data, and set weights/bias layers
'''
def set_threshold(x):
    threshold = np.ptp(x, axis=0).astype(int)
    return threshold

# Set threshold for clip_norm function
threshold = set_threshold(5*series)

# Hyperparameters
epochs = 10000
learning_rate = 1e-5
y_delta = 3 # how far ahead to predict

x_t = np.array([series, np.roll(series, 1, axis=0), np.roll(series, 2, axis=0)]) # go back 1, then 2 steps in time
x_t = np.transpose(x_t, (1, 2, 0)).astype(np.float32) # transpose to [steps, 1, y_delta]

x = x_t[0+y_delta:x_t.shape[0]-y_delta] # truncate to remove rows with shifted values that don't correspond to time
x = np.reshape(x, (x.shape[0], y_delta)) # Reshape 3 dimensional matrix to 2 dimensional matrix

y = np.array(np.roll(series, y_delta, axis=0)).astype(np.float32) # target values are y_delta steps away
y = y[0+y_delta:x_t.shape[0]-y_delta]

print("x shape: {0} y shape: {1}".format(x.shape,y.shape))

# Single layer network with output (Y = W*x + b)
W_layer_1 = np.tanh(np.random.normal(0,size=x.shape)).astype(np.float32)
b_layer_1 = np.zeros((1, W_layer_1.shape[0]), dtype = np.float32)
W_output = np.tanh(np.random.normal(0,size=y.shape)).astype(np.float32)
b_output = np.zeros((1, W_output.shape[0]), dtype = np.float32)

'''
3. Utility functions for neural network
'''

def sigmoid(x):
    clip_norm(x, threshold)
    return .5 * (1 + np.tanh(.5 * x)) # Using a tanh for numerical stability

def gradient(x):
    return x*(1-x)

# Normalizers
def normalize(x):
    x_norm = (x-np.mean(x))/np.std(x)
    return x_norm

def denormalize(x, y):
    y_denorm = y*np.std(x) + np.mean(x)
    return y_denorm

# Min-Max Scalers (alternative to normalize functions)
def scale(x):
    mm = (x - np.min(x))/(np.max(x)-np.min(x))
    return mm

def descale(x, y):
    n = y*(np.max(x)-np.min(x)) + np.min(x)
    return n

# Clip norm of gradient (to prevent exploding gradients)
def clip_norm(x, threshold):
    delta = x*(threshold/np.maximum(x, threshold))
    return delta

'''
4. Run training loop
'''

total_loss = predictions = weights_1 = weights_output = []
for i in range(epochs):
    x_n = normalize(x) # normalize before activation layer
    layer_1 = sigmoid(np.dot(W_layer_1, np.transpose(x_n))) + b_layer_1 # hidden layer
    output = sigmoid(np.dot(np.transpose(W_output), layer_1)) + b_output # predictions
    y_hat = denormalize(x, output) # denormalize to get predicted values

    error = y - y_hat
    loss = np.mean(np.square(error))

    # Print progress
    if (i % 100) == 0:
        print("\n\nEpoch: {0}".format(i))
        print("__________________________________________")
        print("Loss: {0}".format(loss))
        print("Predictions: {0}".format(y_hat[0][-10:-1]))

    # Backprop
    output_delta = error*gradient(output)

    layer_1_delta = output_delta*gradient(layer_1)
    layer_1_error = np.dot(layer_1_delta, W_layer_1)

    # Update weights and bias layers
    W_output += np.dot(output_delta*-learning_rate, W_output)
    W_layer_1 += np.dot(layer_1_delta*-learning_rate, W_layer_1)
    b_output = np.sum(output_delta, axis=0)/output_delta.shape[1]
    b_layer_1 = np.sum(layer_1_delta, axis=0)/layer_1_delta.shape[1]

    if (i % 1000) == 0:
        weights_1.append(W_layer_1)
        weights_output.append(weights_output)

    total_loss.append(loss)
    predictions = y_hat

'''
5. Plot and display results of training
'''

plt.scatter(timeline,series)
plt.show()
print("Total Loss: {0}".format(np.mean(total_loss)))
print("Final Predictions: {0}".format(predictions))
plt.plot(total_loss)
plt.ylabel("loss")
