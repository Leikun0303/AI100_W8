"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='Block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='Block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net


def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24
    compression_rate = 0.5

    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    def Trans_block(net,block_num):
        
        # 1 X 1 X reduce_dim(net) 卷积层
        net = slim.conv2d(net, reduce_dim(net), [1, 1], stride=1,scope=('Conv_trans'+block_num))
        end_points['Conv_trans'+block_num] = net

        # 2 X 2 平均池化层
        net = slim.avg_pool2d(net, [2, 2], stride=2, scope=('Pool_trans'+block_num))
        end_points['Pool_trans'+block_num] = net

        return net

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            #pass
            ##########################
            # Put your code here.

            # 根据论文 Implementation Details. 中描述,图像在进入block之前要进行:
            # 1个 7 X 7 X (2* growth) 的卷积层,步长为2;以及
            # 1个 3 X 3 的最大池化层,步长为2,padding为VALID

            # 7 X 7 X (2* growth)卷积层
            print('images\'s shape:', images.shape)
            net = slim.conv2d(images, 2 * growth, [7, 7], padding='SAME', stride=2, scope='Conv_Before_block')
            end_points['Conv_Before_block'] = net

            # 3 X 3 最大池化层
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='Pool_Before_block')
            end_points['Pool_Before_block'] = net
            
            # 这里采用DenseNet-121的网络结构,即block1为6*2,block2为12*2,block3为24*2,block4为16*2
            # *2表示一个1 X 1 和 3 X 3的卷积层,即函数block()
            # 每个block之后需要进行一次1 X 1,步长为1的卷积层进行压缩和1次2 X 2,步长为2的平均池化层

            print('Conv+pool_Before_block net\'s shape:', net.shape)

            # Dense Block-01
            net = block(net, 6, growth, 'Block01')
            end_points['Block01'] = net

            # Transition Layer-01
            net=Trans_block(net,'01')
        
            print('Block01+Trans01 net\'s shape:', net.shape)

            # Dense Block-02
            net = block(net, 12, growth, 'Block02')
            end_points['Block02'] = net

            # Transition Layer-02
            net=Trans_block(net,'02')

            print('Block02+Trans02 net\'s shape:', net.shape)

            # Dense Block-03
            net = block(net, 24, growth, 'Block03')
            end_points['Block03'] = net

            # Transition Layer-03
            net=Trans_block(net,'03')

            print('Block03+Trans03 net\'s shape:', net.shape)

            # Dense Block-04
            net = block(net, 16, growth, 'Block04')
            end_points['Block04'] = net

            # Transition Layer-04
            net=Trans_block(net,'04')

            print('last layers net\'s shape:', net.shape)
            # 最后一个block04之后,跟着一个 7 X 7的平均池化层
            net = slim.avg_pool2d(net, net.shape[1:3], scope='Pool_global')
            end_points['Pool_global'] = net
            
            # 然后就是将net进行flatten后全连接
            net = slim.flatten(net, scope='Flatten')
            end_points['Flatten'] =net
            
            # 全连接层
            logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='Logits')
            end_points['logits'] = logits
            end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
            
            ##########################

    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
