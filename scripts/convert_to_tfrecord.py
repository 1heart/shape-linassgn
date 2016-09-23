import tensorflow as tf
import numpy as np
import re
from os import listdir, path
from scipy.misc import imread, imshow, imresize

# Utils from tensorflow's convert_to_records
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# TODO:
#       - Don't resize the image
#       - Redo labels (currently, it's ['x-1.gif', 'x-10.gif', ..., 'x-2.gif', 'x-20.gif'])
#       - Generalize labels (currently just 70x20)

if __name__ == '__main__':
    IM_SIZE = (256, 256)
    PATH = '../data/mpeg7/'
    NAME = 'mpeg7_256_256'

    pattern = re.compile('.*-[0-9]+.gif')
    img_names = listdir(PATH); img_names = [f for f in img_names if pattern.match(f)]
    n = len(img_names)

    images = None
    for i in range(len(img_names)):
        if i % 100 == 0: print((str(i * 100.0 / n) + '%'))
        im = imread(PATH+img_names[i], mode='L')
        im = imresize(im, IM_SIZE)
        im = im.flatten()
        if images is None: images = im
        else: images = np.vstack((images, im))

    # The labels: [0, 0, ..., 1, 1, ... 69, 69...]
    labels = np.outer(np.arange(70), np.ones(20)).flatten()

    filename = path.join(PATH, NAME + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(n):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(IM_SIZE[0]),
            'width': _int64_feature(IM_SIZE[1]),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

