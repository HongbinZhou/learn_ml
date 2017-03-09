#!/usr/bin/python3

# thanks: http://www.jeyzhang.com/tensorflow-learning-notes.html


import tensorflow as tf
import numpy as np


m1 = tf.constant(np.array([[1.,2.,3.]]))

m2 = tf.constant(np.array([[4.],[5.],[6.]]))

prod = tf.matmul(m1, m2)

with tf.Session() as sess:
    with tf.device("gpu/:1"):
        result = sess.run(prod)
        print(result)


one = tf.constant(1)
state = tf.Variable(0, name="state")

new_value = tf.add(state, one)

update = tf.assign(state, new_value)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(state))
    for _ in range(3):
        b = sess.run(update)
        print(b)


