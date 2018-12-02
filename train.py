import os
import time
import fnmatch
from random import shuffle
import numpy as np
import tensorflow as tf

from image_ops import get_image, save_images

def train(aacnn):
    """Train AACNN.

    Preconditions:
        checkpoint, data, logs directories exist

    Postconditions:
        checkpoints are saved
        logs are written

    Args:
        aacnn: AACNN object
    """
    sess = aacnn.sess
    FLAGS = aacnn.f

    # load dataset
    list_file = os.path.join(FLAGS.data_dir, '{0}.txt'.format(FLAGS.dataset))
    if os.path.exists(list_file):
        # load from file when found
        print ("Using training list: {0}".format(list_file))
        with open(list_file, 'r') as f:
            data = [os.path.join(FLAGS.data_dir,
                                 FLAGS.dataset, l.strip()) for l in f]
    else:
        # recursively walk dataset directory to get images
        data = []
        dataset_dir = os.path.join(FLAGS.data_dir, FLAGS.dataset)
        for root, dirnames, filenames in os.walk(dataset_dir):
            for filename in fnmatch.filter(filenames, '*.{0}'.format(FLAGS.image_ext)):
                data.append(os.path.join(root, filename))
        shuffle(data)

        # save to file for next time
        with open(list_file, 'w') as f:
            for l in data:
                line = l.replace(dataset_dir + os.sep, '')
                f.write('{0}\n'.format(line))

    assert len(data) > 0, "found 0 training data"
    print ("Found {0} training images.".format(len(data)))

    attribute_data = get_attribute(FLAGS)
    attribute_data = attribute_data.astype(float)
   
    if FLAGS.with_gan:   	 
        learning_rate_decay_d = tf.train.exponential_decay(FLAGS.learning_rate, aacnn.global_step_d,
                                                   3000, 0.8, staircase=True)
    
    learning_rate_decay_g = tf.train.exponential_decay(FLAGS.learning_rate, aacnn.global_step_g,
                                               3000, 0.8, staircase=True)
    # setup RMSProp optimizer
    if FLAGS.with_gan: 
        d_optim = tf.train.RMSPropOptimizer(
            learning_rate=learning_rate_decay_d,
            decay=0.99
            ).minimize(aacnn.d_loss, global_step=aacnn.global_step_d,var_list=aacnn.d_vars)

    g_optim = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate_decay_g,
        decay=0.99
        ).minimize(aacnn.g_loss, global_step=aacnn.global_step_g,var_list=aacnn.g_vars)
   
    tf.global_variables_initializer().run()
    
    # summaries
    if FLAGS.with_gan:
        g_sum = tf.summary.merge([aacnn.d_fake_sum,
                              aacnn.g_sum, aacnn.d_loss_fake_sum, aacnn.g_loss_sum])
        d_sum = tf.summary.merge([aacnn.d_real_sum,
                              aacnn.real_sum, aacnn.d_loss_real_sum, aacnn.d_loss_sum])
    else:
        g_sum = tf.summary.merge([aacnn.g_sum, aacnn.g_loss_sum])
    
    writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, aacnn.get_model_dir()), sess.graph)

    # training images for sampling
    sample_files = data[0:FLAGS.sample_size]
    sample_attributes = attribute_data[0:FLAGS.sample_size]
    sample = [get_image(sample_file,
                        FLAGS.output_size_height, FLAGS.output_size_wight) for sample_file in sample_files]
    sample_images = np.array(sample).astype(np.float32)
    sample_path = os.path.join('./', FLAGS.sample_dir,
                               aacnn.get_model_dir(),
                               'real_samples.png')
    save_images(sample_images, sample_path)

    # run for number of epochs
    counter = 1
    start_time = time.time()
    for epoch in range(FLAGS.epoch):
        num_batches = int(len(data) / FLAGS.batch_size)
        # training iterations
        for batch_index in range(0, num_batches):
            # get batch of images for training
            batch_start = batch_index*FLAGS.batch_size
            batch_end = (batch_index+1)*FLAGS.batch_size
            batch_files = data[batch_start:batch_end]
            batch_labels = attribute_data[batch_start:batch_end]
            batch_images = [get_image(batch_file,
                               FLAGS.output_size_height, FLAGS.output_size_wight) for batch_file in batch_files]

            if FLAGS.with_gan:
                # update D network
                _, summary_str = sess.run([d_optim, d_sum],
                    feed_dict={aacnn.input: batch_images, aacnn.input_attribute: batch_labels})
                writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str = sess.run([g_optim, g_sum],
                    feed_dict={aacnn.input: batch_images, aacnn.input_attribute: batch_labels})
                writer.add_summary(summary_str, counter)
            
            # update G network again for stability
            _, summary_str = sess.run([g_optim, g_sum],
                feed_dict={aacnn.input: batch_images, aacnn.input_attribute: batch_labels})
            writer.add_summary(summary_str, counter)

            # compute errors
            if FLAGS.with_gan:
                errD_fake = aacnn.d_loss_fake.eval({aacnn.input: batch_images, aacnn.input_attribute: batch_labels})
                errD_real = aacnn.d_loss_real.eval({aacnn.input: batch_images, aacnn.input_attribute: batch_labels})
            errG = aacnn.g_loss.eval({aacnn.input: batch_images, aacnn.input_attribute: batch_labels})

            # increment global counter (for saving models)
            counter += 1

            # print stats
            if FLAGS.with_gan:
                print ("[train] epoch: {0}, iter: {1}/{2}, time: {3}, d_loss: {4}, g_loss: {5}".format(
                    epoch, batch_index, num_batches, time.time() - start_time, errD_fake+errD_real, errG))
            else:
                print ("[train] epoch: {0}, iter: {1}/{2}, time: {3}, g_loss: {4}".format(
                    epoch, batch_index, num_batches, time.time() - start_time, errG))

            # sample every 100 iterations
            if np.mod(counter, 100) == 1:
                if FLAGS.with_gan:
                    samples, d_loss, g_loss = aacnn.sess.run(
                        [aacnn.G, aacnn.d_loss, aacnn.g_loss],
                        feed_dict={aacnn.input: sample_images, aacnn.input_attribute: sample_attributes})
                    print ("[sample] time: {0}, d_loss: {1}, g_loss: {2}".format(
                        time.time() - start_time, d_loss, g_loss))
                else:
                    samples, g_loss = aacnn.sess.run(
                        [aacnn.G, aacnn.g_loss],
                        feed_dict={aacnn.input: sample_images, aacnn.input_attribute: sample_attributes})
                    print ("[sample] time: {0}, g_loss: {1}".format(
                        time.time() - start_time, g_loss))
                    
                # save samples for visualization
                sample_path = os.path.join('./', FLAGS.sample_dir,
                                           aacnn.get_model_dir(),
                                           'train_{0:02d}_{1:04d}.png'.format(epoch, batch_index))
                save_images(samples, sample_path)

            # save model every 500 iterations
            if np.mod(counter, 500) == 2:
                aacnn.save(counter)
                print ("[checkpoint] saved: {0}".format(time.time() - start_time))

    # save final model
    aacnn.save(counter)
    print ("[checkpoint] saved: {0}".format(time.time() - start_time))

def get_attribute(FLAGS):
    f = open(FLAGS.label_train_path, 'r')
    labels = []
    while True:
        line = f.readline()
        if line == '':
            break    
        line = line[:len(line)-1].split('\r')
        s = line[0].split()
        labels.append(s)
    labels = np.array(labels).astype(float) 
    return labels


