#!/usr/bin/env python3
import tensorflow as tf

PATHS = {"training_set": "dataset/training_set.tfrecords"}

def load_calibrate(path):

    def record_parse_function(proto):
        record = tf.io.parse_single_example(
            proto,
            {
                "csi": tf.io.FixedLenFeature([], tf.string, default_value=""),
                "pos": tf.io.FixedLenFeature([], tf.string, default_value="")
            },
        )

        csi = tf.ensure_shape(tf.io.parse_tensor(record["csi"], out_type=tf.float32), (8, 8, 8, 64, 2)) # arrays x rows x columns x subcarrier x 2
        csi = tf.complex(csi[..., 0], csi[..., 1])
        csi = tf.signal.fftshift(csi, axes=1)

        position = tf.ensure_shape(tf.io.parse_tensor(record["pos"], out_type=tf.float64), (3))

        return csi, position

    dset = tf.data.TFRecordDataset(path)
    dset = dset.map(record_parse_function, num_parallel_calls = tf.data.AUTOTUNE)

    return dset

training_set = load_calibrate(PATHS["training_set"])

# subsample if less datapoints are desired (M/N * datapoints)
M = 10  # The number of data points you want to keep
N = 10 # The size of the window/cycle
training_set = training_set.enumerate().filter(lambda idx, value : (idx % N < M))
training_set = training_set.map(lambda idx, value : value)