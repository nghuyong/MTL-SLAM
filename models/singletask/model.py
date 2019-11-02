#!/usr/bin/env python
# encoding: utf-8
from models.base.model import BaseModel
import tensorflow as tf
from models.utils import create_rnn_cells, weight_variable, bias_variable, bilstm_layer


class SingleTaskModel(BaseModel):
    def __init__(self, config, model_id):
        super().__init__(model_id, config)

    def get_model_name(self):
        return f'singletask/{self.config.data_set}'

    def _build_model(self):
        self._add_input_layer()
        self._add_meta_encoder()
        self._add_context_encoder()
        self._decoder_layer()
        self._add_loss_layer()
        self._add_train_operation()

    def create_feed_dic(self, batch_data, is_training=True):
        feed_dict = {
            self.keep_prob: 0.5 if is_training else 1.0,
            self.days_input: batch_data['days_input'],
            self.user_input: batch_data['user_input'],
            self.country_input: batch_data['country_input'],
            self.client_input: batch_data['client_input'],
            self.session_input: batch_data['session_input'],
            self.format_input: batch_data['format_input'],
            self.time_input: batch_data['time_input'],
            self.word_input: batch_data['word_input'],
            self.word_embedded: batch_data['word_embedded'],
            self.word_length_input: batch_data['word_length_input'],
            self.char_input: batch_data['char_input'],
            self.char_length_input: batch_data['char_length_input'],
            self.label_input: batch_data['label_input'],
        }
        return feed_dict

    def _add_input_layer(self):
        with tf.name_scope('Inputs'):
            self.keep_prob = tf.placeholder(tf.float32, name='dropout')
            # en_es
            # Meta
            self.days_input = tf.placeholder(tf.float32, [None], name='days_input')
            self.country_input = tf.placeholder(tf.int64, [None], name='country_input')
            self.user_input = tf.placeholder(tf.int64, [None], name='user_input')
            self.client_input = tf.placeholder(tf.int64, [None], name='client_input')
            self.session_input = tf.placeholder(tf.int64, [None], name='session_input')
            self.format_input = tf.placeholder(tf.int64, [None], name='format_input')
            self.time_input = tf.placeholder(tf.float32, [None], name='time_input')
            # Context
            self.word_input = tf.placeholder(tf.int64, [None, None], name='word_input')
            self.word_embedded = tf.placeholder(tf.float32, [None, None, 1024],
                                                name='word_embedded')
            self.word_length_input = tf.placeholder(tf.int64, [None], name='word_length_input')
            self.char_input = tf.placeholder(tf.int64, [None, None], name='char_input')
            self.char_length_input = tf.placeholder(tf.int64, [None, None], name='es_en_char_length_input')
            self.label_input = tf.placeholder(tf.int32, [None, None], name='label_inputs')

    def _add_meta_encoder(self):
        # Embeddings
        user_embedding = tf.get_variable(name='user_embedding',
                                         shape=[self.config.user_size_dic['whole'],
                                                self.config.meta_embedding_dim],
                                         initializer=tf.contrib.layers.xavier_initializer())
        country_embedding = tf.get_variable(name='country_embedding',
                                            shape=[self.config.country_size_dic['whole'],
                                                   self.config.meta_embedding_dim],
                                            initializer=tf.contrib.layers.xavier_initializer())
        client_embedding = tf.get_variable(name='client_embedding',
                                           shape=[self.config.client_size, self.config.meta_embedding_dim],
                                           initializer=tf.contrib.layers.xavier_initializer())
        session_embedding = tf.get_variable(name='session_embedding',
                                            shape=[self.config.session_size, self.config.meta_embedding_dim],
                                            initializer=tf.contrib.layers.xavier_initializer())
        format_embedding = tf.get_variable(name='format_embedding',
                                           shape=[self.config.format_size, self.config.meta_embedding_dim],
                                           initializer=tf.contrib.layers.xavier_initializer())
        with tf.variable_scope('uf_transform'):
            w_uf = tf.get_variable('w', [self.config.context_hidden_dim * 4, self.config.context_hidden_dim * 2])
            b_uf = tf.get_variable('b', [self.config.context_hidden_dim * 2])
        with tf.variable_scope('user_related_mlp'):
            user_related_mlp_w = tf.get_variable('w', [self.config.meta_embedding_dim * 2 + 1,
                                                       self.config.context_hidden_dim * 2])
            user_related_mlp_b = tf.get_variable('b', [self.config.context_hidden_dim * 2])
        with tf.variable_scope('exercise_related_mlp'):
            exercise_related_mlp_w = tf.get_variable('w', [self.config.meta_embedding_dim * 3 + 1,
                                                           self.config.context_hidden_dim * 2])
            exercise_related_mlp_b = tf.get_variable('b', [self.config.context_hidden_dim * 2])

        user_related_info = tf.concat([
            tf.nn.embedding_lookup(user_embedding, self.user_input),
            tf.nn.embedding_lookup(country_embedding, self.country_input),
            tf.expand_dims(self.days_input, -1)
        ], axis=1)
        user_rep = tf.nn.dropout(tf.nn.tanh(
            tf.matmul(user_related_info, user_related_mlp_w) + user_related_mlp_b), self.keep_prob)

        exercise_related_info = tf.concat([
            tf.nn.embedding_lookup(client_embedding, self.client_input),
            tf.nn.embedding_lookup(session_embedding, self.session_input),
            tf.nn.embedding_lookup(format_embedding, self.format_input),
            tf.expand_dims(self.time_input, -1)
        ], axis=1)
        exercise_rep = tf.nn.dropout(tf.nn.tanh(
            tf.matmul(exercise_related_info, exercise_related_mlp_w) + exercise_related_mlp_b), self.keep_prob)
        self.meta_encoder = tf.nn.sigmoid(
            tf.matmul(tf.concat([user_rep, exercise_rep], -1), w_uf) + b_uf)

    def _add_word_level_context_encoder(self):
        # word embedding
        self.word_level_encoder = bilstm_layer(
            inputs=self.word_embedded,
            inputs_length=self.word_length_input,
            hidden_dim=self.config.context_hidden_dim,
            keep_prob=self.keep_prob,
            layers=self.config.word_layers,
            name_scope='word_level_bi_lstm'
        )

    def char_lstm(self, name_scope, char_embedded, char_len):
        with tf.variable_scope(name_scope):
            char_cell = create_rnn_cells(self.config.context_hidden_dim, self.keep_prob)
            outputs, state = tf.nn.dynamic_rnn(
                char_cell, char_embedded, dtype=tf.float32,
                sequence_length=char_len, swap_memory=True)
        char_enc = tf.reduce_sum(outputs, 1) / tf.maximum(
            tf.expand_dims(tf.cast(char_len, tf.float32), 1), 1)
        char_enc = tf.reshape(char_enc, [self.config.batch_size, -1, self.config.context_hidden_dim])
        return char_enc

    def _add_char_level_lstm_context_encoder(self):
        self.chars = tf.reshape(self.char_input, [-1, self.config.char_max_len])
        char_len = tf.reshape(self.char_length_input, [-1, ])
        self.char_embedding = tf.get_variable(name='char_embedding',
                                              shape=[self.config.char_size_dic[self.config.data_set],
                                                     self.config.char_embedding_dim],
                                              initializer=tf.contrib.layers.xavier_initializer())
        char_lstm_encoder = self.char_lstm(name_scope='char_lstm',
                                           char_embedded=tf.nn.embedding_lookup(self.char_embedding,
                                                                                self.chars),
                                           char_len=char_len
                                           )
        self.char_lstm_context_encoder = bilstm_layer(
            inputs=char_lstm_encoder,
            inputs_length=self.word_length_input,
            hidden_dim=self.config.context_hidden_dim,
            keep_prob=self.keep_prob,
            layers=self.config.char_lstm_bilstm_layers,
            name_scope='char_level_lstm_bi_lstm'
        )

    def cnn(self, name_scope, char_embedded):
        char_embedded = tf.expand_dims(char_embedded, -1)
        pooled_outputs = list()
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.variable_scope(f"{name_scope}_conv1_{filter_size}"):
                filter_shape = [filter_size, self.config.char_embedding_dim, 1, self.config.n_filter]
                w_filter = weight_variable(shape=filter_shape, name='w_filter')
                beta = bias_variable(shape=[self.config.n_filter], name='beta_filter')
                conv = tf.nn.bias_add(
                    tf.nn.conv2d(char_embedded, w_filter, strides=[1, 1, 1, 1], padding="VALID",
                                 name="conv"), beta)
                h = tf.nn.relu(conv, name="relu")

            with tf.variable_scope(f"{name_scope}_conv2_{filter_size}"):
                filter_shape = [filter_size, 1, self.config.n_filter, self.config.n_filter]
                w_filter = weight_variable(shape=filter_shape, name='w_filter')
                beta = bias_variable(shape=[self.config.n_filter], name='beta_filter')
                conv = tf.nn.bias_add(
                    tf.nn.conv2d(h, w_filter, strides=[1, 1, 1, 1], padding="VALID", name="conv"),
                    beta)
                h = tf.nn.relu(conv, name="relu")
            pooled = tf.nn.max_pool(h, ksize=[1, self.config.char_max_len - filter_size * 2 + 2, 1, 1],
                                    strides=[1, 1, 1, 1], padding='VALID', name="pool")
            pooled_outputs.append(pooled)
        h_pool = tf.concat(pooled_outputs, 3)
        cnn_char_enc = tf.reshape(h_pool,
                                  [self.config.batch_size, -1,
                                   self.config.n_filter * len(self.config.filter_sizes)])
        return cnn_char_enc

    def _add_char_level_cnn_context_encoder(self):
        char_cnn_encoder = self.cnn(
            name_scope='char_cnn',
            char_embedded=tf.nn.embedding_lookup(self.char_embedding, self.chars)

        )

        self.char_cnn_context_encoder = bilstm_layer(
            inputs=char_cnn_encoder,
            inputs_length=self.word_length_input,
            hidden_dim=self.config.context_hidden_dim,
            keep_prob=self.keep_prob,
            layers=self.config.char_cnn_bilstm_layers,
            name_scope='char_level_cnn_bi_lstm'
        )

    def _add_context_encoder(self):
        self._add_word_level_context_encoder()
        self._add_char_level_lstm_context_encoder()
        self._add_char_level_cnn_context_encoder()

    def _decoder_layer(self):
        with tf.variable_scope('transform'):
            w_trans = tf.get_variable('w_trans',
                                      [self.config.context_hidden_dim * 6, self.config.context_hidden_dim * 2])
            b_trans = tf.get_variable('b_trans', [self.config.context_hidden_dim * 2])

        with tf.variable_scope('proj'):
            w_proj = tf.get_variable('w_proj', [self.config.context_hidden_dim * 2, 1])
            b_proj = tf.get_variable('b_proj', [1, ])
        # en_es
        ta_array = tf.TensorArray(
            dtype=tf.float32,
            size=tf.shape(self.word_input)[1],
            tensor_array_name='outputs_array')
        meta_encoder = self.meta_encoder
        word_level_encoder = self.word_level_encoder
        char_lstm_context_encoder = self.char_lstm_context_encoder
        char_cnn_context_encoder = self.char_cnn_context_encoder

        def cond(step, _):
            return tf.less(step, tf.shape(self.word_input)[1])

        def body(step, ta_out):
            combined = tf.concat(
                [word_level_encoder[:, step, :],
                 char_lstm_context_encoder[:, step, :],
                 char_cnn_context_encoder[:, step, :]
                 ],
                -1)
            trans = tf.nn.sigmoid(tf.matmul(combined, w_trans) + b_trans)
            scr = trans * meta_encoder
            prb = tf.nn.sigmoid(tf.matmul(scr, w_proj) + b_proj)
            ta_out = ta_out.write(step, prb)
            return step + 1, ta_out

        _, preds = tf.while_loop(cond, body, [tf.constant(0), ta_array])
        preds = tf.reshape(
            preds.concat(), [tf.shape(self.word_input)[1],
                             tf.shape(self.word_input)[0]])
        self.y_pred = tf.transpose(preds)

    def _add_loss_layer(self):
        with tf.variable_scope('loss'):
            masked = tf.sequence_mask(self.word_length_input,
                                      tf.reduce_max(self.word_length_input))
            loss = -1 * self.config.pos_weight * (tf.to_float(self.label_input) + 1e-7) * tf.log(
                self.y_pred) - \
                   (1 - self.config.pos_weight) * (1 - tf.to_float(self.label_input)) * tf.log(
                1 - self.y_pred + 1e-7)
            self.loss = tf.reduce_sum(loss * tf.to_float(masked)) / tf.to_float(
                tf.reduce_sum(self.word_length_input))

    def _add_train_operation(self):
        """
        training
        """
        with tf.variable_scope('training_ops'):
            decayed_lr = tf.train.exponential_decay(self.config.lr,
                                                    self.global_step, 10000,
                                                    0.95, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=decayed_lr)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
