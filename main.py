import os
import pprint
pp = pprint.PrettyPrinter()

import tensorflow as tf

from aacnn import AACNN
from train import train
import inference

flags = tf.app.flags
# training params
flags.DEFINE_integer("epoch", 35, "Number of epochs to train. [35]")
flags.DEFINE_float("learning_rate", 0.0005, "Learning rate for Adam optimizer [0.0005]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of Adam optimizer [0.5]")
flags.DEFINE_integer("batch_size", 64, "Number of images in batch [64]")
flags.DEFINE_boolean("with_gan", False, "If jointly use GAN + L2 losses [False]")
flags.DEFINE_integer("l2_weight", 100, "weights for L2 losse [100]")
# model params
flags.DEFINE_integer("output_size_wight", 96, "Size of the output image wight to produce [96]")
flags.DEFINE_integer("output_size_height", 112, "Size of the output image height to produce [112]")
flags.DEFINE_integer("attribute_size", 38, "Size of the input attribute dimension [38]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
# dataset params
flags.DEFINE_string("data_dir", "data", "Path to datasets directory [data]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA]")
# flags for running
flags.DEFINE_string("experiment_name", "experiment", "Name of experiment for current run [experiment]")
flags.DEFINE_boolean("train", False, "Train if True, otherwise test [False]")
flags.DEFINE_integer("sample_size", 64, "Number of images to sample [64]")
# directory or path params
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Path to save the checkpoint data [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Path to save the image samples [samples]")
flags.DEFINE_string("log_dir", "logs", "Path to log for TensorBoard [logs]")
flags.DEFINE_string("image_ext", "jpg", "Image extension to find [jpg]")
flags.DEFINE_string("label_train_path", "label_train.txt", "Path of attribute labels for training [label_train.txt]")
flags.DEFINE_string("label_test_path", "label_test.txt", "Path of attribute labels for testing [label_test.txt]")
flags.DEFINE_string("save_path", "result_attr_result", "Path to save the images when testing  [result_attr_result]")

FLAGS = flags.FLAGS

def main(_):
    #pp.pprint(FLAGS.__flags)

    # training/inference
    with tf.Session() as sess:
        print(FLAGS)
        aacnn = AACNN(sess, FLAGS)
        
        # path checks
        if not os.path.exists(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)
        if not os.path.exists(os.path.join(FLAGS.log_dir, aacnn.get_model_dir())):
            os.makedirs(os.path.join(FLAGS.log_dir, aacnn.get_model_dir()))
        if not os.path.exists(os.path.join(FLAGS.sample_dir, aacnn.get_model_dir())):
            os.makedirs(os.path.join(FLAGS.sample_dir, aacnn.get_model_dir()))
        if not os.path.exists(FLAGS.save_path):
            os.makedirs(FLAGS.save_path)
        if not os.path.exists(os.path.join(FLAGS.save_path, 'gt')):
            os.makedirs(os.path.join(FLAGS.save_path, 'gt'))

        # load checkpoint if found
        if aacnn.checkpoint_exists():
            print("Loading checkpoints...")
            if aacnn.load():
                print ("success!")
            else:
                raise IOError("Could not read checkpoints from {0}!".format(
                    FLAGS.checkpoint_dir))
        else:
            if not FLAGS.train:
                raise IOError("No checkpoints found but need for sampling!")
            print ("No checkpoints found. Training from scratch.")
            aacnn.load()

        # train AACNN
        if FLAGS.train:
            train(aacnn)

        # inference/visualization code goes here
        print ("Generating samples...")
        inference.generate_image(aacnn)

if __name__ == '__main__':
    tf.app.run()
