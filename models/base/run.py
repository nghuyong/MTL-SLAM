#!/usr/bin/env python
# encoding: utf-8
import os

import argparse
import tensorflow as tf


class BaseRun(object):
    def __init__(self):
        self.ARGS = None
        self.parser_init()
        self.tf_init()

    def parser_init(self):
        parser = argparse.ArgumentParser(description='Train global model')
        parser.add_argument('train_or_test', nargs='?', help='choose train or test model', choices=['train', 'test'],
                            default='train')
        parser.add_argument('--gpu', help="gpu device", default='4')
        parser.add_argument('--model_id', help="model id", default='0')
        parser.add_argument('--model_ids', help="model ids", default='0,1,2,3')
        parser.add_argument('--data_set', help="data_set", default='en_es')
        self.ARGS = parser.parse_args()

    def tf_init(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.ARGS.gpu)
        tf.logging.set_verbosity(tf.logging.ERROR)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        return sess

    def trainer_init(self):
        NotImplementedError

    def model_init(self):
        NotImplementedError

    def run(self):
        sess = self.tf_init()
        trainer = self.trainer_init()
        model = self.model_init()
        if self.ARGS.train_or_test == 'train':
            sess.run(tf.global_variables_initializer())
            trainer.train(model, sess)
        else:
            model_ids = [int(item) for item in self.ARGS.model_ids.split(',')]
            trainer.restore_and_test_model(model, sess, model_ids=model_ids)
