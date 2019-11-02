#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf
from tensorflow.contrib import rnn


def weight_variable(shape, name):
    """create W"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    """create bias"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def create_rnn_cells(hid_size, keep_prob, cell_num=1,
                     forget_bias=1.0, use_residual=True, is_return_list=False):
    cell_list = []
    for i in range(cell_num):
        cell = tf.nn.rnn_cell.LSTMCell(
            hid_size, forget_bias=forget_bias)
        cell = tf.contrib.rnn.DropoutWrapper(
            cell=cell, input_keep_prob=keep_prob)
        if use_residual:
            cell = tf.contrib.rnn.ResidualWrapper(cell)
        cell_list.append(cell)
    if cell_num == 1:
        return cell_list[0]
    if is_return_list:
        return cell_list
    else:
        return tf.contrib.rnn.MultiRNNCell(cell_list)


def lstm_cell(hidden_size, scope, keep_prob):
    with tf.name_scope(scope):
        cell = rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
        cell = rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell


def bilstm_layer(inputs, inputs_length, hidden_dim, keep_prob, layers, name_scope):
    with tf.variable_scope(name_scope):
        fw_cell = create_rnn_cells(
            hidden_dim, keep_prob, layers, use_residual=False)
        bw_cell = create_rnn_cells(
            hidden_dim, keep_prob, layers, use_residual=False)
        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, inputs, dtype=tf.float32,
            sequence_length=inputs_length)
        return tf.concat(bi_outputs, -1)
