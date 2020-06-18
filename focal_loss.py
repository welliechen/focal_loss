#coding=utf-8
"""
@Author: chenhongliang
@Date:   2020-05-21
@Desc:
    Tensorflow实现何凯明的Focal Loss, 该损失函数主要用于解决分类问题中的类别不平衡
    focal_loss_sigmoid: 二分类loss
    focal_loss_softmax: 多分类loss
    Reference Paper : Focal Loss for Dense Object Detection
"""


import tensorflow as tf

def focal_loss_sigmoid(labels,logits,alpha=0.25,gamma=2):
    """
    Computer focal loss for binary classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter. If positive samples number
      > negtive samples number, alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred = tf.nn.sigmoid(logits)
    labels = tf.to_float(labels)

    L = -labels*(1-alpha)*((1-y_pred)**gamma)*tf.log(y_pred)-\
      (1-labels)*alpha*(y_pred**gamma)*tf.log(1-y_pred)

    return L

def focal_loss_softmax(labels, logits, alphas, gamma=2):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      alphas: A float array like [alpha_c1, alpha_c2, ..., alpha_cN] and num_classes is the length.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    # softmax 归一
    y_pred = tf.nn.softmax(logits, dim=-1) # [batch_size,num_classes]

    # sparse表示转成one-hot表示
    labels = tf.squeeze(labels)
    labels_new = tf.one_hot(labels, y_pred.shape[1])#[n, c]

    # 生成alpha对角阵
    alphas = tf.constant(alphas, dtype=tf.float32)
    alphas_matrix = tf.matrix_diag(alphas)

    # 增加alpha变量
    L = -tf.matmul(labels_new, alphas_matrix)*((1-y_pred)**gamma)*tf.log(y_pred) #[n, c]
    # 没有alpha变量
    #L = -labels*((1-y_pred)**gamma)*tf.log(y_pred) #[n, c]
    L = tf.reduce_sum(L, axis=1) #[n]

    return L, labels, y_pred

def ce_loss_softmax(labels, logits):
    # 交叉熵分为两步 1）求softmax，softmax = tf.nn.softmax(logits)；  2）对得到的列表每个元素求对数再求相反数得到
    # labels : [batch_size], logits : [batch_size,num_classes]
    L = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, reduction=tf.losses.Reduction.NONE)
    return L


if __name__ == '__main__':
    import os
    import sys
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    logits=tf.random_uniform(shape=[5],minval=-1,maxval=1,dtype=tf.float32)
    labels=tf.Variable([0,1,0,0,1])
    loss1=focal_loss_sigmoid(labels=labels,logits=logits)

    #logits2=tf.random_uniform(shape=[5,4],minval=-1,maxval=1,dtype=tf.float32)
    logits2 = tf.Variable([[0.1, 0.9], [0.4, 0.6], [0.2, 0.8], [0.2, 0.7]])

    labels2=tf.Variable([1,0,1,1])
    #labels2 = tf.Variable([0, 1, 0, 0, 1])
    loss2, labels, preds = focal_loss_softmax(labels=labels2, logits=logits2, alphas=[3, 1], gamma=0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('focal loss binary:')
        print (sess.run(loss1))
        print("labels:")
        print(sess.run(labels))
        print("preds:")
        print(sess.run(preds))
        print('focal loss result')
        print (sess.run(loss2))
        #print(sess.run(loss3))
        print('ce_loss:')
        import math
        print('---  验证crossEntroy正确性  ---')
        print(-math.log(0.6899744)*1, -math.log(0.450166)*1, -math.log(0.6456563), -math.log(0.62245935))
        print(sess.run(ce_loss_softmax(labels2, logits2)))


