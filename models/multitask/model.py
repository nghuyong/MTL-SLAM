#!/usr/bin/env python
# encoding: utf-8
from models.base.model import BaseModel
import tensorflow as tf
from models.utils import create_rnn_cells, weight_variable, bias_variable, bilstm_layer


class MultiTaskModel(BaseModel):
    def __init__(self, config, model_id):
        super().__init__(model_id, config)

    def get_model_name(self):
        return 'multitask'

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

            self.en_es_days_input: batch_data['en_es']['days_input'],
            self.en_es_user_input: batch_data['en_es']['user_input'],
            self.en_es_country_input: batch_data['en_es']['country_input'],
            self.en_es_client_input: batch_data['en_es']['client_input'],
            self.en_es_session_input: batch_data['en_es']['session_input'],
            self.en_es_format_input: batch_data['en_es']['format_input'],
            self.en_es_time_input: batch_data['en_es']['time_input'],
            self.en_es_word_embedded: batch_data['en_es']['word_embedded'],
            self.en_es_word_input: batch_data['en_es']['word_input'],
            self.en_es_word_length_input: batch_data['en_es']['word_length_input'],
            self.en_es_char_input: batch_data['en_es']['char_input'],
            self.en_es_char_length_input: batch_data['en_es']['char_length_input'],
            self.en_es_label_input: batch_data['en_es']['label_input'],

            self.es_en_days_input: batch_data['es_en']['days_input'],
            self.es_en_user_input: batch_data['es_en']['user_input'],
            self.es_en_country_input: batch_data['es_en']['country_input'],
            self.es_en_client_input: batch_data['es_en']['client_input'],
            self.es_en_session_input: batch_data['es_en']['session_input'],
            self.es_en_format_input: batch_data['es_en']['format_input'],
            self.es_en_time_input: batch_data['es_en']['time_input'],
            self.es_en_word_embedded: batch_data['es_en']['word_embedded'],
            self.es_en_word_input: batch_data['es_en']['word_input'],
            self.es_en_word_length_input: batch_data['es_en']['word_length_input'],
            self.es_en_char_input: batch_data['es_en']['char_input'],
            self.es_en_char_length_input: batch_data['es_en']['char_length_input'],
            self.es_en_label_input: batch_data['es_en']['label_input'],

            self.fr_en_days_input: batch_data['fr_en']['days_input'],
            self.fr_en_user_input: batch_data['fr_en']['user_input'],
            self.fr_en_country_input: batch_data['fr_en']['country_input'],
            self.fr_en_client_input: batch_data['fr_en']['client_input'],
            self.fr_en_session_input: batch_data['fr_en']['session_input'],
            self.fr_en_format_input: batch_data['fr_en']['format_input'],
            self.fr_en_time_input: batch_data['fr_en']['time_input'],
            self.fr_en_word_embedded: batch_data['fr_en']['word_embedded'],
            self.fr_en_word_input: batch_data['fr_en']['word_input'],
            self.fr_en_word_length_input: batch_data['fr_en']['word_length_input'],
            self.fr_en_char_input: batch_data['fr_en']['char_input'],
            self.fr_en_char_length_input: batch_data['fr_en']['char_length_input'],
            self.fr_en_label_input: batch_data['fr_en']['label_input'],

        }
        return feed_dict

    def _add_input_layer(self):
        with tf.name_scope('Inputs'):
            self.keep_prob = tf.placeholder(tf.float32, name='dropout')
            # en_es
            # Meta
            self.en_es_days_input = tf.placeholder(tf.float32, [None], name='en_es_days_input')
            self.en_es_country_input = tf.placeholder(tf.int64, [None], name='en_es_country_input')
            self.en_es_user_input = tf.placeholder(tf.int64, [None], name='en_es_user_input')
            self.en_es_client_input = tf.placeholder(tf.int64, [None], name='en_es_client_input')
            self.en_es_session_input = tf.placeholder(tf.int64, [None], name='en_es_session_input')
            self.en_es_format_input = tf.placeholder(tf.int64, [None], name='en_es_format_input')
            self.en_es_time_input = tf.placeholder(tf.float32, [None], name='en_es_time_input')
            # Context
            self.en_es_word_embedded = tf.placeholder(tf.float32, [None, None, 1024], name='en_es_word_embedded')
            self.en_es_word_input = tf.placeholder(tf.int64, [None, None], name='en_es_word_input')
            self.en_es_word_length_input = tf.placeholder(tf.int64, [None], name='en_es_word_length_input')
            self.en_es_char_input = tf.placeholder(tf.int64, [None, None], name='en_es_char_input')
            self.en_es_char_length_input = tf.placeholder(tf.int64, [None, None], name='es_en_char_length_input')
            self.en_es_label_input = tf.placeholder(tf.int32, [None, None], name='en_es_label_inputs')

            # es_en
            # Meta
            self.es_en_days_input = tf.placeholder(tf.float32, [None], name='es_en_days_input')
            self.es_en_country_input = tf.placeholder(tf.int64, [None], name='es_en_country_input')
            self.es_en_user_input = tf.placeholder(tf.int64, [None], name='es_en_user_input')
            self.es_en_client_input = tf.placeholder(tf.int64, [None], name='es_en_client_input')
            self.es_en_session_input = tf.placeholder(tf.int64, [None], name='es_en_session_input')
            self.es_en_format_input = tf.placeholder(tf.int64, [None], name='es_en_format_input')
            self.es_en_time_input = tf.placeholder(tf.float32, [None], name='es_en_time_input')
            # Context
            self.es_en_word_embedded = tf.placeholder(tf.float32, [None, None, 1024], name='es_en_word_embedded')
            self.es_en_word_input = tf.placeholder(tf.int64, [None, None], name='es_en_word_input')
            self.es_en_word_length_input = tf.placeholder(tf.int64, [None], name='es_en_word_length_input')
            self.es_en_char_input = tf.placeholder(tf.int64, [None, None], name='es_en_char_input')
            self.es_en_char_length_input = tf.placeholder(tf.int64, [None, None], name='es_en_char_length_input')
            self.es_en_label_input = tf.placeholder(tf.int32, [None, None], name='es_en_label_inputs')

            # fr_en
            # Meta
            self.fr_en_days_input = tf.placeholder(tf.float32, [None], name='fr_en_days_input')
            self.fr_en_country_input = tf.placeholder(tf.int64, [None], name='fr_en_country_input')
            self.fr_en_user_input = tf.placeholder(tf.int64, [None], name='fr_en_user_input')
            self.fr_en_client_input = tf.placeholder(tf.int64, [None], name='fr_en_client_input')
            self.fr_en_session_input = tf.placeholder(tf.int64, [None], name='fr_en_session_input')
            self.fr_en_format_input = tf.placeholder(tf.int64, [None], name='fr_en_format_input')
            self.fr_en_time_input = tf.placeholder(tf.float32, [None], name='fr_en_time_input')
            # Context
            self.fr_en_word_embedded = tf.placeholder(tf.float32, [None, None, 1024], name='fr_en_word_embedded')
            self.fr_en_word_input = tf.placeholder(tf.int64, [None, None], name='fr_en_word_input')
            self.fr_en_word_length_input = tf.placeholder(tf.int64, [None], name='fr_en_word_length_input')
            self.fr_en_char_input = tf.placeholder(tf.int64, [None, None], name='fr_en_char_input')
            self.fr_en_char_length_input = tf.placeholder(tf.int64, [None, None], name='fr_en_char_length_input')
            self.fr_en_label_input = tf.placeholder(tf.int32, [None, None], name='fr_en_label_inputs')

    def _add_meta_encoder(self):
        # Embeddings
        user_embedding = tf.get_variable(name='user_embedding',
                                         shape=[self.config.user_size_dic['whole'], self.config.meta_embedding_dim],
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
        # en_es
        en_es_user_related_info = tf.concat([
            tf.nn.embedding_lookup(user_embedding, self.en_es_user_input),
            tf.nn.embedding_lookup(country_embedding, self.en_es_country_input),
            tf.expand_dims(self.en_es_days_input, -1)
        ], axis=1)
        en_es_user_rep = tf.nn.dropout(tf.nn.tanh(
            tf.matmul(en_es_user_related_info, user_related_mlp_w) + user_related_mlp_b), self.keep_prob)

        en_es_exercise_related_info = tf.concat([
            tf.nn.embedding_lookup(client_embedding, self.en_es_client_input),
            tf.nn.embedding_lookup(session_embedding, self.en_es_session_input),
            tf.nn.embedding_lookup(format_embedding, self.en_es_format_input),
            tf.expand_dims(self.en_es_time_input, -1)
        ], axis=1)
        en_es_exercise_rep = tf.nn.dropout(tf.nn.tanh(
            tf.matmul(en_es_exercise_related_info, exercise_related_mlp_w) + exercise_related_mlp_b), self.keep_prob)
        self.en_es_meta_encoder = tf.nn.sigmoid(
            tf.matmul(tf.concat([en_es_user_rep, en_es_exercise_rep], -1), w_uf) + b_uf)

        # es_en
        es_en_user_related_info = tf.concat([
            tf.nn.embedding_lookup(user_embedding, self.es_en_user_input),
            tf.nn.embedding_lookup(country_embedding, self.es_en_country_input),
            tf.expand_dims(self.es_en_days_input, -1)
        ], axis=1)
        es_en_user_rep = tf.nn.dropout(tf.nn.tanh(
            tf.matmul(es_en_user_related_info, user_related_mlp_w) + user_related_mlp_b), self.keep_prob)
        es_en_exercise_related_info = tf.concat([
            tf.nn.embedding_lookup(client_embedding, self.es_en_client_input),
            tf.nn.embedding_lookup(session_embedding, self.es_en_session_input),
            tf.nn.embedding_lookup(format_embedding, self.es_en_format_input),
            tf.expand_dims(self.es_en_time_input, -1)
        ], axis=1)
        es_en_exercise_rep = tf.nn.dropout(tf.nn.tanh(
            tf.matmul(es_en_exercise_related_info, exercise_related_mlp_w) + exercise_related_mlp_b), self.keep_prob)

        self.es_en_meta_encoder = tf.nn.sigmoid(
            tf.matmul(tf.concat([es_en_user_rep, es_en_exercise_rep], -1), w_uf) + b_uf)

        # fr_en
        fr_en_user_related_info = tf.concat([
            tf.nn.embedding_lookup(user_embedding, self.fr_en_user_input),
            tf.nn.embedding_lookup(country_embedding, self.fr_en_country_input),
            tf.expand_dims(self.fr_en_days_input, -1)
        ], axis=1)
        fr_en_user_rep = tf.nn.dropout(tf.nn.tanh(
            tf.matmul(fr_en_user_related_info, user_related_mlp_w) + user_related_mlp_b), self.keep_prob)
        fr_en_exercise_related_info = tf.concat([
            tf.nn.embedding_lookup(client_embedding, self.fr_en_client_input),
            tf.nn.embedding_lookup(session_embedding, self.fr_en_session_input),
            tf.nn.embedding_lookup(format_embedding, self.fr_en_format_input),
            tf.expand_dims(self.fr_en_time_input, -1)
        ], axis=1)
        fr_en_exercise_rep = tf.nn.dropout(tf.nn.tanh(
            tf.matmul(fr_en_exercise_related_info, exercise_related_mlp_w) + exercise_related_mlp_b), self.keep_prob)

        self.fr_en_meta_encoder = tf.nn.sigmoid(
            tf.matmul(tf.concat([fr_en_user_rep, fr_en_exercise_rep], -1), w_uf) + b_uf)

    def _add_word_level_context_encoder(self):
        # en_es
        self.en_es_word_level_encoder = bilstm_layer(
            inputs=self.en_es_word_embedded,
            inputs_length=self.en_es_word_length_input,
            hidden_dim=self.config.context_hidden_dim,
            keep_prob=self.keep_prob,
            layers=self.config.word_layers,
            name_scope='en_es_word_level_bi_lstm'
        )

        # es_en
        self.es_en_word_level_encoder = bilstm_layer(
            inputs=self.es_en_word_embedded,
            inputs_length=self.es_en_word_length_input,
            hidden_dim=self.config.context_hidden_dim,
            keep_prob=self.keep_prob,
            layers=self.config.word_layers,
            name_scope='es_en_word_level_bi_lstm'
        )

        # fr_en
        self.fr_en_word_level_encoder = bilstm_layer(
            inputs=self.fr_en_word_embedded,
            inputs_length=self.fr_en_word_length_input,
            hidden_dim=self.config.context_hidden_dim,
            keep_prob=self.keep_prob,
            layers=self.config.word_layers,
            name_scope='fr_en_word_level_bi_lstm'
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
        # en_es
        self.en_es_chars = tf.reshape(self.en_es_char_input, [-1, self.config.char_max_len])
        en_es_char_len = tf.reshape(self.en_es_char_length_input, [-1, ])
        self.en_es_char_embedding = tf.get_variable(name='en_es_char_embedding',
                                                    shape=[self.config.char_size_dic['en_es'],
                                                           self.config.char_embedding_dim],
                                                    initializer=tf.contrib.layers.xavier_initializer())
        en_es_char_lstm_encoder = self.char_lstm(name_scope='en_es_char_lstm',
                                                 char_embedded=tf.nn.embedding_lookup(self.en_es_char_embedding,
                                                                                      self.en_es_chars),
                                                 char_len=en_es_char_len
                                                 )
        self.en_es_char_lstm_context_encoder = bilstm_layer(
            inputs=en_es_char_lstm_encoder,
            inputs_length=self.en_es_word_length_input,
            hidden_dim=self.config.context_hidden_dim,
            keep_prob=self.keep_prob,
            layers=self.config.char_lstm_bilstm_layers,
            name_scope='en_es_char_level_lstm_bi_lstm'
        )

        # es_en
        self.es_en_chars = tf.reshape(self.es_en_char_input, [-1, self.config.char_max_len])
        es_en_char_len = tf.reshape(self.es_en_char_length_input, [-1, ])
        self.es_en_char_embedding = tf.get_variable(name='es_en_char_embedding',
                                                    shape=[self.config.char_size_dic['es_en'],
                                                           self.config.char_embedding_dim],
                                                    initializer=tf.contrib.layers.xavier_initializer())
        es_en_char_lstm_encoder = self.char_lstm(name_scope='es_en_char_lstm',
                                                 char_embedded=tf.nn.embedding_lookup(self.es_en_char_embedding,
                                                                                      self.es_en_chars),
                                                 char_len=es_en_char_len
                                                 )
        self.es_en_char_lstm_context_encoder = bilstm_layer(
            inputs=es_en_char_lstm_encoder,
            inputs_length=self.es_en_word_length_input,
            hidden_dim=self.config.context_hidden_dim,
            keep_prob=self.keep_prob,
            layers=self.config.char_lstm_bilstm_layers,
            name_scope='es_en_char_level_lstm_bi_lstm'
        )

        # fr_en
        self.fr_en_chars = tf.reshape(self.fr_en_char_input, [-1, self.config.char_max_len])
        fr_en_char_len = tf.reshape(self.fr_en_char_length_input, [-1, ])
        self.fr_en_char_embedding = tf.get_variable(name='fr_en_char_embedding',
                                                    shape=[self.config.char_size_dic['fr_en'],
                                                           self.config.char_embedding_dim],
                                                    initializer=tf.contrib.layers.xavier_initializer())
        fr_en_char_lstm_encoder = self.char_lstm(name_scope='fr_en_char_lstm',
                                                 char_embedded=tf.nn.embedding_lookup(self.fr_en_char_embedding,
                                                                                      self.fr_en_chars),
                                                 char_len=fr_en_char_len
                                                 )
        self.fr_en_char_lstm_context_encoder = bilstm_layer(
            inputs=fr_en_char_lstm_encoder,
            inputs_length=self.fr_en_word_length_input,
            hidden_dim=self.config.context_hidden_dim,
            keep_prob=self.keep_prob,
            layers=self.config.char_lstm_bilstm_layers,
            name_scope='fr_en_char_level_lstm_bi_lstm'
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
        # en_es
        en_es_char_cnn_encoder = self.cnn(
            name_scope='en_es_char_cnn',
            char_embedded=tf.nn.embedding_lookup(self.en_es_char_embedding, self.en_es_chars)

        )

        self.en_es_char_cnn_context_encoder = bilstm_layer(
            inputs=en_es_char_cnn_encoder,
            inputs_length=self.en_es_word_length_input,
            hidden_dim=self.config.context_hidden_dim,
            keep_prob=self.keep_prob,
            layers=self.config.char_cnn_bilstm_layers,
            name_scope='en_es_char_level_cnn_bi_lstm'
        )

        # es_en
        es_en_char_cnn_encoder = self.cnn(
            name_scope='es_en_char_cnn',
            char_embedded=tf.nn.embedding_lookup(self.es_en_char_embedding, self.es_en_chars)

        )
        self.es_en_char_cnn_context_encoder = bilstm_layer(
            inputs=es_en_char_cnn_encoder,
            inputs_length=self.es_en_word_length_input,
            hidden_dim=self.config.context_hidden_dim,
            keep_prob=self.keep_prob,
            layers=self.config.char_cnn_bilstm_layers,
            name_scope='es_en_char_level_cnn_bi_lstm'
        )

        # fr_en
        fr_en_char_cnn_encoder = self.cnn(
            name_scope='fr_en_char_cnn',
            char_embedded=tf.nn.embedding_lookup(self.fr_en_char_embedding, self.fr_en_chars)

        )
        self.fr_en_char_cnn_context_encoder = bilstm_layer(
            inputs=fr_en_char_cnn_encoder,
            inputs_length=self.fr_en_word_length_input,
            hidden_dim=self.config.context_hidden_dim,
            keep_prob=self.keep_prob,
            layers=self.config.char_cnn_bilstm_layers,
            name_scope='fr_en_char_level_cnn_bi_lstm'
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
        en_es_ta_array = tf.TensorArray(
            dtype=tf.float32,
            size=tf.shape(self.en_es_word_input)[1],
            tensor_array_name='en_es_outputs_array')
        en_es_meta_encoder = self.en_es_meta_encoder
        en_es_word_level_encoder = self.en_es_word_level_encoder
        en_es_char_lstm_context_encoder = self.en_es_char_lstm_context_encoder
        en_es_char_cnn_context_encoder = self.en_es_char_cnn_context_encoder

        def en_es_cond(step, _):
            return tf.less(step, tf.shape(self.en_es_word_input)[1])

        def en_es_body(step, ta_out):
            combined = tf.concat(
                [en_es_word_level_encoder[:, step, :],
                 en_es_char_lstm_context_encoder[:, step, :],
                 en_es_char_cnn_context_encoder[:, step, :]
                 ],
                -1)
            trans = tf.nn.sigmoid(tf.matmul(combined, w_trans) + b_trans)
            scr = trans * en_es_meta_encoder
            prb = tf.nn.sigmoid(tf.matmul(scr, w_proj) + b_proj)
            ta_out = ta_out.write(step, prb)
            return step + 1, ta_out

        en_es_, en_es_preds = tf.while_loop(en_es_cond, en_es_body, [tf.constant(0), en_es_ta_array])
        en_es_preds = tf.reshape(
            en_es_preds.concat(), [tf.shape(self.en_es_word_input)[1],
                                   tf.shape(self.en_es_word_input)[0]])
        self.en_es_y_pred = tf.transpose(en_es_preds)

        # es_en
        es_en_ta_array = tf.TensorArray(
            dtype=tf.float32,
            size=tf.shape(self.es_en_word_input)[1],
            tensor_array_name='es_en_outputs_array')
        es_en_meta_encoder = self.es_en_meta_encoder
        es_en_word_level_encoder = self.es_en_word_level_encoder
        es_en_char_lstm_context_encoder = self.es_en_char_lstm_context_encoder
        es_en_char_cnn_context_encoder = self.es_en_char_cnn_context_encoder

        def es_en_cond(step, _):
            return tf.less(step, tf.shape(self.es_en_word_input)[1])

        def es_en_body(step, ta_out):
            combined = tf.concat(
                [es_en_word_level_encoder[:, step, :],
                 es_en_char_lstm_context_encoder[:, step, :],
                 es_en_char_cnn_context_encoder[:, step, :]
                 ],
                -1)
            trans = tf.nn.sigmoid(tf.matmul(combined, w_trans) + b_trans)
            scr = trans * es_en_meta_encoder
            prb = tf.nn.sigmoid(tf.matmul(scr, w_proj) + b_proj)
            ta_out = ta_out.write(step, prb)
            return step + 1, ta_out

        es_en_, es_en_preds = tf.while_loop(es_en_cond, es_en_body, [tf.constant(0), es_en_ta_array])
        es_en_preds = tf.reshape(
            es_en_preds.concat(), [tf.shape(self.es_en_word_input)[1],
                                   tf.shape(self.es_en_word_input)[0]])
        self.es_en_y_pred = tf.transpose(es_en_preds)

        # fr_en
        fr_en_ta_array = tf.TensorArray(
            dtype=tf.float32,
            size=tf.shape(self.fr_en_word_input)[1],
            tensor_array_name='fr_en_outputs_array')
        fr_en_meta_encoder = self.fr_en_meta_encoder
        fr_en_word_level_encoder = self.fr_en_word_level_encoder
        fr_en_char_lstm_context_encoder = self.fr_en_char_lstm_context_encoder
        fr_en_char_cnn_context_encoder = self.fr_en_char_cnn_context_encoder

        def fr_en_cond(step, _):
            return tf.less(step, tf.shape(self.fr_en_word_input)[1])

        def fr_en_body(step, ta_out):
            combined = tf.concat(
                [fr_en_word_level_encoder[:, step, :],
                 fr_en_char_lstm_context_encoder[:, step, :],
                 fr_en_char_cnn_context_encoder[:, step, :]
                 ],
                -1)
            trans = tf.nn.sigmoid(tf.matmul(combined, w_trans) + b_trans)
            scr = trans * fr_en_meta_encoder
            prb = tf.nn.sigmoid(tf.matmul(scr, w_proj) + b_proj)
            ta_out = ta_out.write(step, prb)
            return step + 1, ta_out

        fr_en_, fr_en_preds = tf.while_loop(fr_en_cond, fr_en_body, [tf.constant(0), fr_en_ta_array])
        fr_en_preds = tf.reshape(
            fr_en_preds.concat(), [tf.shape(self.fr_en_word_input)[1],
                                   tf.shape(self.fr_en_word_input)[0]])
        self.fr_en_y_pred = tf.transpose(fr_en_preds)

    def _add_loss_layer(self):
        # en_es
        with tf.variable_scope('en_es_loss'):
            en_es_masked = tf.sequence_mask(self.en_es_word_length_input,
                                            tf.reduce_max(self.en_es_word_length_input))
            en_es_loss = -1 * self.config.pos_weight * (tf.to_float(self.en_es_label_input) + 1e-7) * tf.log(
                self.en_es_y_pred) - \
                         (1 - self.config.pos_weight) * (1 - tf.to_float(self.en_es_label_input)) * tf.log(
                1 - self.en_es_y_pred + 1e-7)
            self.en_es_loss = tf.reduce_sum(en_es_loss * tf.to_float(en_es_masked)) / tf.to_float(
                tf.reduce_sum(self.en_es_word_length_input))
        # es_en
        with tf.variable_scope('es_en_loss'):
            es_en_masked = tf.sequence_mask(self.es_en_word_length_input,
                                            tf.reduce_max(self.es_en_word_length_input))
            es_en_loss = -1 * self.config.pos_weight * (tf.to_float(self.es_en_label_input) + 1e-7) * tf.log(
                self.es_en_y_pred) - \
                         (1 - self.config.pos_weight) * (1 - tf.to_float(self.es_en_label_input)) * tf.log(
                1 - self.es_en_y_pred + 1e-7)
            self.es_en_loss = tf.reduce_sum(es_en_loss * tf.to_float(es_en_masked)) / tf.to_float(
                tf.reduce_sum(self.es_en_word_length_input))
        # fr_en
        with tf.variable_scope('fr_en_loss'):
            fr_en_masked = tf.sequence_mask(self.fr_en_word_length_input,
                                            tf.reduce_max(self.fr_en_word_length_input))
            fr_en_loss = -1 * self.config.pos_weight * (tf.to_float(self.fr_en_label_input) + 1e-7) * tf.log(
                self.fr_en_y_pred) - \
                         (1 - self.config.pos_weight) * (1 - tf.to_float(self.fr_en_label_input)) * tf.log(
                1 - self.fr_en_y_pred + 1e-7)
            self.fr_en_loss = tf.reduce_sum(fr_en_loss * tf.to_float(fr_en_masked)) / tf.to_float(
                tf.reduce_sum(self.fr_en_word_length_input))

        self.loss = self.en_es_loss + self.es_en_loss + self.fr_en_loss

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

