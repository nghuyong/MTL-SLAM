#!/usr/bin/env python
# encoding: utf-8
import pickle
import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self, is_training, char_max_len=15):
        self.is_training = is_training
        self.char_max_len = char_max_len

    def get_data_path(self, data_set):
        if self.is_training:
            return f'./data/processed/{data_set}/train_and_dev.csv'
        else:
            return f'./data/processed/{data_set}/test.csv'

    @staticmethod
    def load_meta_embedding():
        with open(f'./data/processed/embedding/whole_countries.txt', 'rt', encoding='utf-8') as f:
            country2id = {value: index for index, value in enumerate(f.read().splitlines())}

        with open(f'./data/processed/embedding/whole_users.txt', 'rt', encoding='utf-8') as f:
            user2id = {value: index for index, value in enumerate(f.read().splitlines())}

        with open('./data/processed/embedding/client.txt', 'rt', encoding='utf-8') as f:
            client2id = {value: index for index, value in enumerate(f.read().splitlines())}

        with open('./data/processed/embedding/format.txt', 'rt', encoding='utf-8') as f:
            format2id = {value: index for index, value in enumerate(f.read().splitlines())}

        with open('./data/processed/embedding/session.txt', 'rt', encoding='utf-8') as f:
            session2id = {value: index for index, value in enumerate(f.read().splitlines())}

        return country2id, user2id, client2id, format2id, session2id

    @staticmethod
    def load_context_embedding(data_set):
        with open(f'./data/processed/embedding/{data_set}_elmo.pkl', 'rb') as f:
            word_embedded_dic = pickle.load(f)

        with open(f'./data/processed/embedding/{data_set}_words.txt', 'rt', encoding='utf-8') as f:
            word2id = {value: index for index, value in enumerate(f.read().splitlines())}

        with open(f'./data/processed/embedding/{data_set}_characters.txt', 'rt', encoding='utf-8') as f:
            char2id = {value: index for index, value in enumerate(f.read().splitlines())}
        return word2id, word_embedded_dic, char2id

    @staticmethod
    def prepare_one_batch_meta_data(one_batch_data, user2id, country2id, client2id, session2id, format2id):
        """
        one_batch_data: pandas data frame
        """
        batch_size = len(one_batch_data)
        user_input = np.zeros(batch_size)
        country_input = np.zeros(batch_size)
        days_input = np.zeros(batch_size)
        client_input = np.zeros(batch_size)
        session_input = np.zeros(batch_size)
        format_input = np.zeros(batch_size)
        time_input = np.zeros(batch_size)
        min_index = one_batch_data.index.min()
        for index in range(batch_size):
            one_row_data = one_batch_data.loc[min_index + index]
            # user
            user_input[index] = user2id[one_row_data['user']]
            # country
            country_input[index] = country2id[one_row_data['country'].split('|')[0]]
            # days
            days_input[index] = float(one_row_data['days'])
            # client
            client_input[index] = client2id[one_row_data['client']]
            # session
            session_input[index] = session2id[one_row_data['session']]
            # format
            format_input[index] = format2id[one_row_data['format']]
            # time
            if float(one_row_data['time']) >= 100:
                time_input[index] = 100
            elif float(one_row_data['time']) < 1:
                time_input[index] = 0
            else:
                time_input[index] = one_row_data['time']
        return {
            'country_input': country_input,
            'days_input': days_input,
            'client_input': client_input,
            'session_input': session_input,
            'format_input': format_input,
            'time_input': time_input,
            'user_input': user_input,
        }

    @staticmethod
    def prepare_one_batch_context_and_label_data(one_batch_data, char_max_len, word2id, word_embedded_dic, char2id):
        batch_size = len(one_batch_data)
        sentence_lengths = [len(_.split()) for _ in one_batch_data['sentence']]
        max_sentence_length = max(sentence_lengths)
        label_input = np.zeros((batch_size, max_sentence_length))
        word_input = np.zeros((batch_size, max_sentence_length))
        word_embedded = np.zeros((batch_size, max_sentence_length, 1024))
        char_length_input = np.zeros((batch_size, max_sentence_length))
        char_input = np.zeros((batch_size, char_max_len * max_sentence_length))
        word_length_input = np.array(sentence_lengths)
        min_index = one_batch_data.index.min()
        for index in range(batch_size):
            one_row_data = one_batch_data.loc[min_index + index]
            raw_words = one_row_data['sentence'].lower().split()
            word_ids = [word2id.get(word, word2id.get('UNK')) for word in raw_words]
            word_input[index, :len(word_ids)] += np.array(word_ids)
            char_length_input[index, :len(word_ids)] += np.array([len(word) for word in raw_words])
            word_embedded[index, :len(raw_words), :] += word_embedded_dic[one_row_data['sentence'].lower()]
            for word_index, word in enumerate(raw_words):
                chars = list(word)
                chars_id = [char2id.get(char.lower(), char2id.get('UNK')) for char in chars]
                word_start = word_index * char_max_len
                char_input[index, word_start:word_start + len(chars_id)] += np.array(chars_id)
            label = [int(_) for _ in one_row_data['label'].split()]
            label_input[index, :len(label)] += np.array(label)
        return {
            'word_embedded': word_embedded,
            'word_input': word_input,
            'char_length_input': char_length_input,
            'char_input': char_input,
            'word_length_input': word_length_input,
            'label_input': label_input
        }

    def batch_data_generator(self, data_path, batch_size):
        data_set = pd.read_csv(data_path, index_col=False, header=None,
                               names=['user', 'sentence', 'label', 'country', 'days', 'client', 'session', 'format',
                                      'time'])
        while True:
            if self.is_training:
                data_set = data_set.sample(frac=1).reset_index(drop=True)
            start_index = 0
            while True:
                if start_index + batch_size > len(data_set):
                    break
                one_batch_data = data_set[start_index:start_index + batch_size]
                yield one_batch_data
                start_index += batch_size
            if not self.is_training:
                break
