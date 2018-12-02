import tensorflow as tf
import tensorflow.contrib.slim as slim

class Generator():
    """Generator model.
    """
    def __init__(self, FLAGS):
        """Initialization.

        Args:
            FLAGS: flags object
        """
        self.f = FLAGS

    def __call__(self, image, attribute, reuse=False):
        """Generator function call for training.

        Args:
            image: input LR image (batch size, width, height, 3)
            attribute: input attribute (batch size, width, height, 3)
            reuse: reuse variables [False]

        Returns:
            generated image
                parameterized in range of [-1, 1]
                dimensionality is (batch size, width, height, 3)
        """
        # downsample HR image 8x to LR image
        pool1 = slim.avg_pool2d(image, [2, 2])
        pool2 = slim.avg_pool2d(pool1, [2, 2])
        pool3 = slim.avg_pool2d(pool2, [2, 2])
      
        prelu_alpha = 0.25
        # Architecture of Feature Extractor Branch A
        with tf.name_scope('g/ba_conv1') as scope:
            reuse_scope = scope if reuse else None
            ba_conv1 = slim.conv2d(pool3, 32, [3, 3],
                stride=1,
                activation_fn=tf.contrib.keras.layers.PReLU(
                    alpha_initializer=tf.constant_initializer(prelu_alpha)),
                reuse=reuse_scope,
                scope='g/ba_conv1')

        with tf.name_scope('g/ba_conv2') as scope:
            reuse_scope = scope if reuse else None
            ba_conv2 = slim.conv2d(ba_conv1, 32, [3, 3],
                stride=1,
                activation_fn=tf.contrib.keras.layers.PReLU(
                    alpha_initializer=tf.constant_initializer(prelu_alpha)),
                reuse=reuse_scope,
                scope='g/ba_conv2')

        with tf.name_scope('g/ba_conv3') as scope:
            reuse_scope = scope if reuse else None
            ba_conv3 = slim.conv2d(ba_conv2, 32, [3, 3],
                stride=1,
                activation_fn=None,
                reuse=reuse_scope,
                scope='g/ba_conv3')

        # Architecture of Feature Extractor Branch B
        with tf.name_scope('g/bb_fc1') as scope:
            bb_fc1 = slim.fully_connected(attribute, 504,
                    activation_fn=tf.contrib.keras.layers.PReLU(
                        alpha_initializer=tf.constant_initializer(prelu_alpha)),
                    reuse=reuse,
                    scope='g/bb_fc1')
            bb_fc1_reshape = tf.reshape(bb_fc1, [-1, 14, 12, 3])
        
        with tf.name_scope('g/bb_conv1') as scope:
            reuse_scope = scope if reuse else None
            bb_conv1 = slim.conv2d(bb_fc1_reshape, 32, [3, 3],
                stride=1,
                activation_fn=tf.contrib.keras.layers.PReLU(
                    alpha_initializer=tf.constant_initializer(prelu_alpha)),
                reuse=reuse_scope,
                scope='g/bb_conv1')
        
        with tf.name_scope('g/bb_conv2') as scope:
            reuse_scope = scope if reuse else None
            bb_conv2 = slim.conv2d(bb_conv1, 32, [3, 3],
                stride=1,
                activation_fn=tf.contrib.keras.layers.PReLU(
                    alpha_initializer=tf.constant_initializer(prelu_alpha)),
                reuse=reuse_scope,
                scope='g/bb_conv2')

        with tf.name_scope('g/bb_conv3') as scope:
            reuse_scope = scope if reuse else None
            bb_conv3 = slim.conv2d(bb_conv2, 32, [3, 3],
                stride=1,
                activation_fn=None,
                reuse=reuse_scope,
                scope='g/bb_conv3')

        # Architecture of Feature Extractor

        fe_concat = tf.concat([ba_conv3, bb_conv3], 3)

        with tf.name_scope('g/fe_preconv1') as scope:
            reuse_scope = scope if reuse else None
            fe_preconv1 = slim.conv2d(fe_concat, 32, [3, 3],
                stride=1,
                activation_fn=None,
                reuse=reuse_scope,
                scope='g/fe_preconv1')

        with tf.name_scope('g/fe_preconv2') as scope:
            reuse_scope = scope if reuse else None
            fe_preconv2 = slim.conv2d(fe_preconv1, 32, [3, 3],
                stride=1,
                activation_fn=tf.contrib.keras.layers.PReLU(
                    alpha_initializer=tf.constant_initializer(prelu_alpha)),
                reuse=reuse_scope,
                scope='g/fe_preconv2')

        with tf.name_scope('g/fe_preconv3') as scope:
            reuse_scope = scope if reuse else None
            fe_preconv3 = slim.conv2d(fe_preconv2, 32, [3, 3],
                stride=1,
                activation_fn=tf.contrib.keras.layers.PReLU(
                    alpha_initializer=tf.constant_initializer(prelu_alpha)),
                reuse=reuse_scope,
                scope='g/fe_preconv3')

        with tf.name_scope('g/fe_preconv4') as scope:
            reuse_scope = scope if reuse else None
            fe_preconv4 = slim.conv2d(fe_preconv3, 32, [3, 3],
                stride=1,
                activation_fn=tf.contrib.keras.layers.PReLU(
                    alpha_initializer=tf.constant_initializer(prelu_alpha)),
                reuse=reuse_scope,
                scope='g/fe_preconv4')

        with tf.name_scope('g/fe_deconv1') as scope:
            reuse_scope = scope if reuse else None
            fe_deconv1 = slim.conv2d_transpose(fe_preconv4, 64, [3, 3],
                stride=2,
                activation_fn=None,
                reuse=reuse_scope,
                scope='g/fe_deconv1')

        fe_deconv1_act =  tf.contrib.keras.layers.PReLU(
                    alpha_initializer=tf.constant_initializer(prelu_alpha))(fe_deconv1) 
   

        with tf.name_scope('g/fe_deconv2') as scope:
            reuse_scope = scope if reuse else None
            fe_deconv2 = slim.conv2d_transpose(fe_deconv1_act, 64, [3, 3],
                stride=2,
                activation_fn=None,
                reuse=reuse_scope,
                scope='g/fe_deconv2')

        fe_deconv2_act =  tf.contrib.keras.layers.PReLU(
                    alpha_initializer=tf.constant_initializer(prelu_alpha))(fe_deconv2) 

                
        with tf.name_scope('g/fe_deconv3') as scope:
            reuse_scope = scope if reuse else None
            fe_deconv3 = slim.conv2d_transpose(fe_deconv2_act, 128, [5, 5],
                stride=2,
                activation_fn=None,
                reuse=reuse_scope,
                scope='g/fe_deconv3')
 
        # Architecture of Generator
        with tf.name_scope('g/preconv1') as scope:
            reuse_scope = scope if reuse else None
            preconv1 = slim.conv2d_transpose(pool3, 512, [3, 3],
                stride=1,
                activation_fn=None,
                reuse=reuse_scope,
                scope='g/preconv1')
            preconv1 = tf.concat([preconv1, fe_preconv1], 3)

        preconv1_act =  tf.contrib.keras.layers.PReLU(
                    alpha_initializer=tf.constant_initializer(prelu_alpha))(preconv1) 
 

        with tf.name_scope('g/deconv1') as scope:
            reuse_scope = scope if reuse else None
            deconv1 = slim.conv2d_transpose(preconv1_act, 256, [3, 3],
                stride=2,
                activation_fn=None,
                reuse=reuse_scope,
                scope='g/deconv1')
            deconv1 = tf.concat([deconv1, fe_deconv1], 3)

        deconv1_act =  tf.contrib.keras.layers.PReLU(
                    alpha_initializer=tf.constant_initializer(prelu_alpha))(deconv1) 
  
        with tf.name_scope('g/deconv2') as scope:
            reuse_scope = scope if reuse else None
            deconv2 = slim.conv2d_transpose(deconv1_act, 128, [5, 5],
                stride=2,
                activation_fn=None,
                reuse=reuse_scope,
                scope='g/deconv2')
            deconv2 = tf.concat([deconv2, fe_deconv2], 3)

        deconv2_act =  tf.contrib.keras.layers.PReLU(
                    alpha_initializer=tf.constant_initializer(prelu_alpha))(deconv2) 
   
        with tf.name_scope('g/deconv3') as scope:
            reuse_scope = scope if reuse else None
            deconv3 = slim.conv2d_transpose(deconv2_act, 64, [5, 5],
                stride=2,
                activation_fn=None,
                reuse=reuse_scope,
                scope='g/deconv3')
            deconv3 = tf.concat([deconv3, fe_deconv3], 3)

        deconv3_act =  tf.contrib.keras.layers.PReLU(
                    alpha_initializer=tf.constant_initializer(prelu_alpha))(deconv3) 
    
        with tf.name_scope('g/recon') as scope:
            reuse_scope = scope if reuse else None
            recon = slim.conv2d_transpose(deconv3_act, 3, [5, 5],
                stride=1,
                activation_fn=None,
                reuse=reuse_scope,
                scope='g/recon')

        return tf.nn.tanh(recon)
