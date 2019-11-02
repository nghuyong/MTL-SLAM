#!/usr/bin/env python
# encoding: utf-8
from models.base.config import BaseConfig


class MultiTaskConfig(BaseConfig):
    batch_size = 128
    lr = 1e-3
    pos_weight = 0.7
    # Meta Encoder
    meta_embedding_dim = 150
    user_related_MLP_layers = 1
    exercise_related_MLP_layers = 1

    # Context Encoder
    context_hidden_dim = 150
    # Word level
    word_embedding_dim = 150
    word_layers = 2

    # Char level LSTM
    char_layers = 3
    char_max_len = 15
    char_embedding_dim = 150
    char_lstm_bilstm_layers = 3

    # Char level CNN
    filter_sizes = [2, 3, 4, 5]
    n_filter = 256
    char_cnn_bilstm_layers = 3