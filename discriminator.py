import tensorflow as tf
import tensorflow.contrib.slim as slim

def lrelu(x, leak=0.2):
    """Leaky ReLU activation.

    Args:
        x: input
        leak: leak parameter [0.2]

    Returns:
        activation
    """
    return tf.maximum(x, leak*x)

class Discriminator():
    """Discriminator model.
    """
    def __init__(self, FLAGS):
        """Initialization.

        Args:
            FLAGS: flags object
        """
        self.f = FLAGS

    def __call__(self, image, reuse=False):
        """Discriminator function call for training.

        Args:
            image: input image (batch size, width, height, 3)
            reuse: reuse variables [False]

        Returns:
            2-tuple:
                1. probability of real image
                    values in range of [0, 1]
                    dimensionality is (batch size, 1)

                2. logits instead of probability
        """
        with tf.name_scope('d/h0') as scope:
            net = slim.conv2d(image, 32, [5, 5],
                    stride=1,
                    normalizer_fn=slim.batch_norm,
                    activation_fn=lrelu,
                    reuse=reuse,
                    scope='d/h0')
        net = slim.max_pool2d(net, [2, 2])
        
        with tf.name_scope('d/h1') as scope:
            net = slim.conv2d(net, 64, [5, 5],
                    stride=1,
                    normalizer_fn=slim.batch_norm,
                    activation_fn=lrelu,
                    reuse=reuse,
                    scope='d/h1')
        net = slim.max_pool2d(net, [2, 2])

        with tf.name_scope('d/h2') as scope:
            net = slim.conv2d(net, 128, [5, 5],
                    stride=1,
                    normalizer_fn=slim.batch_norm,
                    activation_fn=lrelu,
                    reuse=reuse,
                    scope='d/h2')
        net = slim.max_pool2d(net, [2, 2])
        
        with tf.name_scope('d/h3') as scope:
            net = slim.conv2d(net, 96, [5, 5],
                    stride=1,
                    normalizer_fn=slim.batch_norm,
                    activation_fn=lrelu,
                    reuse=reuse,
                    scope='d/h3')
        net = slim.max_pool2d(net, [2, 2])     

        net_shape = net.get_shape()
        extra_dim = net_shape[1].value*net_shape[2].value*net_shape[3].value    
        net = tf.reshape(net, [-1, extra_dim])
        
        with tf.name_scope('d/h4') as scope:
            net = slim.fully_connected(net, 1024,
                    activation_fn=lrelu,
                    reuse=reuse,
                    scope='d/h4')
        
        with tf.name_scope('d/h5') as scope:
            net = slim.fully_connected(net, 1,
                    activation_fn=None,
                    reuse=reuse,
                    scope='d/h5')
        
        return tf.nn.sigmoid(net), net
