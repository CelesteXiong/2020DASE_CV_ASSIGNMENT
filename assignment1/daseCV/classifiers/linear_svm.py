from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += X[i] # dW计算
                dW[:,y[i]] += -X[i] # dW计算

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W) 

    #############################################################################
    # TODO：
    # 计算损失函数的梯度并将其存储为dW。
    # 与其先计算损失再计算梯度，还不如在计算损失的同时计算梯度更简单。
    # 因此，您可能需要修改上面的一些代码来计算梯度。
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW = dW / num_train
    dW += 2 * reg * W
#     pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO: 
    # 实现一个向量化SVM损失计算方法,并将结果存储到loss中
    #############################################################################
    
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0]
    scores = X.dot(W) # (500, 1)
    correct_cls_scores = scores[np.arange(num_train), y]
    correct_cls_scores = np.reshape(correct_cls_scores, (num_train,-1)) # (500, 1)
    
    delta = scores - correct_cls_scores + 1
    delta[np.arange(num_train), y] = 0 

    loss_matrix = np.maximum(0, delta)
    # print(loss_matrix.shape) # (500, 10)

    loss = np.sum(loss_matrix) / num_train + reg * np.sum(W * W) 
    
    
    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                              
    # 实现一个向量化的梯度计算方法,并将结果存储到dW中                                
    #                                                                           
    # 提示:与其从头计算梯度,不如利用一些计算loss时的中间变量                                    
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    '''
     求w_{i}{j}的梯度的思路: 
     1. i≠y_j时: 
     - 对于图片k,若其损失L_{k}≠0,  w_{i}{j}只对s_{k}{j}有贡献, 即L_{k}对w_{i}{j}求导, 得x_{k}{i}
     - 所以用所有图片的损失值中的非零的部分, 即Σ_{k=1, L_{k}≠0}{num_train} L_{k}
     - 对w_{i}{j}求导, 得Σ{k=1, L_{k}≠0}{num_train} x_{k}{i} 
     2. i＝y_j时, 将1.中的 x_{k}{i} 改为: x_{k}{i} * i≠y_j时max()大于0的位置的个数
    '''
    
    # 根据梯度公式, 需要计算上游梯度
    # 1. 对于每一张图片(delta中的每一行) i≠y_j时大于0的位置对应的w_i的梯度为x_i, 等于0时梯度为0
    # 2. i＝y_j时大于0的位置对应的w_i的梯度为 Σ(-x_i), 等于0时梯度为0 [此处的Σ理解为,i≠y_j时大于0的位置的个数]
    
    # 将i≠y_j时max()值大于0的位置的上的值置1
    mask = np.zeros(delta.shape)
    mask[delta > 0] = 1 
    
    # 将i＝y_j时max()大于0的位置的上的值置为i≠y_j时max()大于0的位置的个数, 并且取负
    count = np.sum(mask, axis=1)
    mask[np.arange(num_train),y] = - count
    # print(delta.shape) # (500, 10)
    
    # 用点积形式, 将上游梯度(即处理后的delta)和本地梯度(即X)相乘, 得到W的梯度
    dW = (X.T).dot(mask)
    dW = dW / num_train
    dW += 2 * reg * W
    
    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
