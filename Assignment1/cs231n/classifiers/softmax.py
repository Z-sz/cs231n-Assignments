import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  D, C = W.shape
  N = X.shape[0]
  a = np.dot(X, W)
  row_max = np.max(a, axis=1, keepdims=True)
  a_trim = a - row_max #for numerical stability
  exp = np.exp(a_trim)
  y_hat = exp / np.sum(exp, axis=1, keepdims=True)
  for i in range(N):
    loss -= np.log(y_hat[i, y[i]])
  loss /= N
  loss += reg * np.sum(np.square(W)) / 2
  da = np.zeros(a.shape)
  for i in range(N):
    for j in range(C):
      if(j == y[i]):
        da[i, j] = y_hat[i, j] - 1
      else:
        da[i, j] = y_hat[i, j]
  dW = np.dot(X.T, da) / N + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  a = np.dot(X, W)
  row_max = np.max(a, axis=1, keepdims=True)
  a_trim = a - row_max #for numerical stability
  exp = np.exp(a_trim)
  y_hat = exp / np.sum(exp, axis=1, keepdims=True)
  one_hot = np.zeros_like(y_hat)
  one_hot[range(N), y] = 1
  loss = -np.sum(np.log(y_hat[range(N), y])) / N + reg * np.sum(np.square(W)) / 2
  da = (y_hat - one_hot) / N
  dW = np.dot(X.T, da) + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

