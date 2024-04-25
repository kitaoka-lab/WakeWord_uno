import tensorflow as tf
import numpy as np
from pathlib import Path

THRESHOLD = 0.505
#THRESHOLD = 0.65
NUM_PHONEMES = 6

label = np.load('data/' + str(NUM_PHONEMES) + '/eval/y.npy')
pred = np.load('data/' + str(NUM_PHONEMES) + '/eval/p.npy')

label = tf.one_hot(tf.cast(label, tf.int64), 2)

pred = pred * [0, 1]
pred = pred + [THRESHOLD, 0]
pred = tf.one_hot(tf.math.argmax(
    pred, axis=2, output_type=tf.int64), 2)

label_negative = label * [1, 0]
label_negative = label_negative[:, :, ::-1]
pred_positive = pred * [0, 1]
label_negative_sum = tf.math.reduce_sum(label_negative).numpy() + 1e-07
false_positive_sum = tf.math.reduce_sum(
    label_negative * pred_positive).numpy()

print(label_negative_sum)
print(false_positive_sum)
