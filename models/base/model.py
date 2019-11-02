#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf
import os
import logging


class BaseModel(object):

    def __init__(self, model_id, config):
        self.max_auroc = -1.0
        self.max_f1 = 0.0
        self.config = config
        self.model_id = model_id
        self.model_name = self.get_model_name()
        self.model_path = f'{self.model_name}/{model_id}/'
        self.log_dir = './logs/' + self.model_path
        self.checkpoint_dir = './checkpoints/' + self.model_path
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self._init_logger()
        self.logger.info('=' * 20)
        self.logger.info(f'model name {self.model_name},model id {model_id}')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.logger.info('building model...')
        self._build_model()
        self.logger.info('finish build model')
        self.saver = tf.train.Saver(max_to_keep=None)

    def get_model_name(self):
        NotImplementedError

    def _init_logger(self):
        self.logger = logging.getLogger('model')
        self.logger.setLevel(logging.DEBUG)
        log_format = logging.Formatter(
            fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        file_handler = logging.FileHandler(self.log_dir + 'log.txt', mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)  # 输出到file的log等级的开关
        file_handler.setFormatter(log_format)
        self.logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # 创建该handler的formatter
        console_handler.setFormatter(log_format)
        self.logger.addHandler(console_handler)

    def _build_model(self):
        raise NotImplementedError

    def create_feed_dic(self, batch_data):
        raise NotImplementedError
