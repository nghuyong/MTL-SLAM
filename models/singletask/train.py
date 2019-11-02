#!/usr/bin/env python
# encoding: utf-8
import os

from utils.metric import calculate_metric
import logging
import tensorflow as tf


class SingleTaskTrainer(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.max_result = {'auc': 0.0, 'f1': 0.0}
        self.not_update_count = 0
        self.global_step_for_test_dic = {
            'en_es': 6000,
            'fr_en': 2000,
            'es_en': 2000
        }

    def check_is_early_stop(self):
        # 均连续5轮没有更新，就提前终止训练
        if self.not_update_count > 5:
            return True
        else:
            return False

    def train(self, model, sess):
        logger = logging.getLogger('model')
        logger.info('==========start training===========')
        global_step_for_test = self.global_step_for_test_dic[model.config.data_set]
        train_batch_generator = self.data_loader(is_training=True).generate_train_batch_data(model.config.batch_size,
                                                                                             model.config.data_set)
        logger = logging.getLogger('model')
        logger.info(f'global_step_for_test is {global_step_for_test}')
        while True:
            for batch in train_batch_generator:
                feed_dict = model.create_feed_dic(batch)
                _, global_step = sess.run(
                    [model.train_op, model.global_step],
                    feed_dict)
                if global_step % global_step_for_test == 0:
                    logger.info('=' * 20)
                    logger.info(f'global_step {global_step} start test')
                    self.test_one_data_set(model, sess, global_step)
                    if self.check_is_early_stop():
                        logger.info('early stop!!!')
                        exit(0)

    def test_one_data_set(self, model, sess, global_step=0, is_restore_test=False):
        logger = logging.getLogger('model')
        self.not_update_count += 1
        test_data_loader = self.data_loader(is_training=False).generate_test_batch_data(
            model.config.batch_size, model.config.data_set)
        y_pred_list = []
        y_true_list = []
        for batch in test_data_loader:
            feed_dict = model.create_feed_dic(batch, is_training=False)
            y_pred, global_step = sess.run([model.y_pred, model.global_step], feed_dict)
            for i in range(model.config.batch_size):
                y_pred_list.extend(list(y_pred[i, :batch['word_length_input'][i]]))
                y_true_list.extend(
                    list(batch['label_input'][i, :batch['word_length_input'][i]]))
        metric = calculate_metric(y_true_list, y_pred_list)
        auc, f1 = metric['auroc'], metric['F1']
        logger.info(f'test result auc {auc} f1 {f1}')
        if is_restore_test:
            return y_true_list, y_pred_list
        if auc > self.max_result['auc'] or (
                auc == self.max_result['auc'] and f1 > self.max_result['f1']):
            logger.info(f'find new best model!!!')
            self.max_result['auc'] = auc
            self.max_result['f1'] = f1
            self.not_update_count = 0
            os.system(f'rm -r {model.checkpoint_dir}/')
            os.makedirs(f'{model.checkpoint_dir}/')
            model.saver.save(sess, f'{model.checkpoint_dir}{global_step}_{auc}_{f1}.ckpt')
            logger.info(f'save into {model.checkpoint_dir}{global_step}_{auc}_{f1}.ckpt')
        logger.info(str(self.max_result))

    def restore_and_test_model(self, model, sess, model_ids):
        def list_add(a, b):
            c = []
            for i in range(len(a)):
                c.append(a[i] + b[i])
            return c

        logger = logging.getLogger('model')
        logger.info('restore and  test model')
        sum_y_pred_list = None
        sum_y_true_list = None
        for model_id in model_ids:
            checkpoint_path = f'./checkpoints/{model.model_name}/{model_id}/'
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(checkpoint_path)
            saver.restore(sess, ckpt)
            y_true_list, y_pred_list = self.test_one_data_set(model, sess, is_restore_test=True)
            if not sum_y_pred_list:
                sum_y_pred_list = y_pred_list
                sum_y_true_list = y_true_list
            else:
                assert len(sum_y_pred_list) == len(y_pred_list)
                assert sum_y_true_list == y_true_list
                sum_y_pred_list = list_add(sum_y_pred_list, y_pred_list)
        sum_y_pred_list = [_ / len(model_ids) for _ in sum_y_pred_list]
        metric = calculate_metric(sum_y_true_list, sum_y_pred_list)
        logger.info('final result: ' + str(metric))
