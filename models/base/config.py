#!/usr/bin/env python
# encoding: utf-8

class BaseConfig(object):
    word_size_dic = {
        'fr_en': 1521,
        'en_es': 1801,
        'es_en': 2209,
    }
    char_size_dic = {
        'fr_en': 43,
        'en_es': 33,
        'es_en': 35,
    }
    user_size_dic = {
        'fr_en': 1213,
        'en_es': 2593,
        'es_en': 2643,
        'whole': 6447
    }
    country_size_dic = {
        'fr_en': 93,
        'en_es': 37,
        'es_en': 106,
        'whole': 126
    }
    train_data_size_dic = {
        'fr_en': 326792,
        'en_es': 824012,
        'es_en': 731896,
    }
    dev_data_size_dic = {
        'fr_en': 43610,
        'en_es': 115770,
        'es_en': 96003,
    }
    test_data_size_dic = {
        'fr_en': 41753,
        'en_es': 114586,
        'es_en': 93145,
    }
    format_size = 3
    session_size = 3
    client_size = 3
