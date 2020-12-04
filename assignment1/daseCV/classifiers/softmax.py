from builtins import range
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
    # TODO: 使用显式循环计算softmax损失及其梯度。
    # 将损失和梯度分别保存在loss和dW中。
    # 如果你不小心，很容易遇到数值不稳定的情况。 
    # 不要忘了正则化！                                                           
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)
        score_sum = np.sum(np.exp(scores))
        
        for j in range(num_classes):
            if j == y[i]:
                loss += -np.log(np.exp(scores[j]) / score_sum)
                dW[:, j] += (np.exp(scores[j]) / score_sum - 1) * X[i]
            else:
                dW[:, j] += (np.exp(scores[j]) / score_sum) * X[i]
                
    loss = loss / num_train + reg * np.sum(W*W)
    dW = dW / num_train + 2 * reg * W
    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # TODO: 不使用显式循环计算softmax损失及其梯度。
    # 将损失和梯度分别保存在loss和dW中。
    # 如果你不小心，很容易遇到数值不稳定的情况。 
    # 不要忘了正则化！
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_c = W.shape[1]
    scores = X.dot(W)
    score_sum = np.sum(np.exp(scores), axis = 1)
    # print(score_sum.shape)
    p_vector = np.exp(scores[np.arange(num_train), y]) / score_sum
    loss = np.sum(-np.log(p_vector)) / num_train + reg * np.sum(W*W) 
                      
    # create mask
    mask = np.zeros(scores.shape)
    sum_i = np.sum(np.exp(scores[np.arange(scores.shape[0])]), axis=1)
    # print(sum_i.shape) (N,)
    temp = np.exp(scores[np.arange(scores.shape[0])]) / np.reshape(sum_i, [sum_i.shape[0], -1])
    # print(temp.shape) (N, C)
    mask[np.arange(mask.shape[0])] = np.reshape(temp,[temp.shape[0], -1])
    mask[np.arange(mask.shape[0]), y] -= 1
    # print(mask.shape) (N, C)
    
    # calculate gradient
    dW = X.T.dot(mask) 
    dW = dW / num_train + reg * 2 * W
                      
    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
