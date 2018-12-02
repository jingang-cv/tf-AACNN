import os
import numpy as np
from time import gmtime, strftime

from image_ops import get_image, save_image, save_images

def generate_image(aacnn):
    """

    Postconditions:
        saves to image file

    Args:
        aacnn: AACNN
        output_file: path to save image file [test.png] 
    """
    FLAGS = aacnn.f
    # load dataset
    list_file = os.path.join(FLAGS.data_dir, 'Img_test.txt')
    #list_file = 'attribute_testset/img_19.txt'
    if os.path.exists(list_file):
        # load from file when found
        print ("Using training list: {0}".format(list_file))
        with open(list_file, 'r') as f:
            data = [os.path.join(FLAGS.data_dir,
                                 FLAGS.dataset, l.strip()) for l in f]
            #data = [l.strip()  for l in f]
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

    num_batches = int(len(data) / FLAGS.batch_size)
    # testing iterations
    count = 1
    for batch_index in range(0, num_batches):
        # get batch of images for testing
        batch_start = batch_index*FLAGS.batch_size
        batch_end = (batch_index+1)*FLAGS.batch_size
        batch_files = data[batch_start:batch_end]
        batch_labels = attribute_data[batch_start:batch_end]
        batch_images = [get_image(batch_file,
                           FLAGS.output_size_height, FLAGS.output_size_wight) for batch_file in batch_files]
        samples = aacnn.sess.run(aacnn.G, feed_dict={aacnn.input: batch_images, aacnn.input_attribute: batch_labels})
        for i in range(FLAGS.batch_size):
           save_image(samples[i], os.join.path(FLAGS.save_path, str(count) + '.jpg'))
           save_image(batch_images[i], os.join.path(FLAGS.save_path, 'gt', str(count) + '.jpg'))
           count += 1

def get_attribute(FLAGS):
    f = open(FLAGS.label_test_path, 'r')
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


