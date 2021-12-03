import tensorflow as tf


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
    return feature


def create_byte_feature(values):
    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[values.numpy()]))
    return feature
