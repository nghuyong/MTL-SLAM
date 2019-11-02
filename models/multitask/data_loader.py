#!/usr/bin/env python
# encoding: utf-8
from models.base.data_loader import DataLoader


class MultiTaskDataLoader(DataLoader):
    def __init__(self, is_training):
        super().__init__(is_training)

    def generate_train_batch_data(self, one_data_set_batch_size):
        data_sets = ['en_es', 'es_en', 'fr_en']
        data_generator_dic = {}
        context_embedding_dic = {}
        for data_set in data_sets:
            data_generator_dic[data_set] = self.batch_data_generator(self.get_data_path(data_set),
                                                                     one_data_set_batch_size)
            context_embedding_dic[data_set] = self.load_context_embedding(data_set)
        country2id, user2id, client2id, format2id, session2id = self.load_meta_embedding()
        while True:
            yield_batch_data = {}
            for data_set in data_sets:
                one_batch_data = data_generator_dic[data_set].__next__()

                prepared_batch_data = self.prepare_one_batch_meta_data(one_batch_data, user2id, country2id,
                                                                       client2id, session2id, format2id)
                word2id, word_embedded_dic, char2id = context_embedding_dic[data_set]
                prepared_batch_data.update(
                    self.prepare_one_batch_context_and_label_data(one_batch_data, self.char_max_len, word2id,
                                                                  word_embedded_dic, char2id))
                yield_batch_data[data_set] = prepared_batch_data
            yield yield_batch_data

    def generate_test_batch_data(self, one_data_set_batch_size, test_data_set):
        data_sets = ['en_es', 'es_en', 'fr_en']
        data_generator_dic = {}
        context_embedding_dic = {}
        for data_set in data_sets:
            data_generator_dic[data_set] = self.batch_data_generator(self.get_data_path(data_set),
                                                                     one_data_set_batch_size)
            context_embedding_dic[data_set] = self.load_context_embedding(data_set)
        country2id, user2id, client2id, format2id, session2id = self.load_meta_embedding()
        yield_batch_data = {}
        for data_set in data_sets:
            one_batch_data = data_generator_dic[data_set].__next__()
            prepared_batch_data = self.prepare_one_batch_meta_data(one_batch_data, user2id, country2id,
                                                                   client2id, session2id, format2id)
            word2id, word_embedded_dic, char2id = context_embedding_dic[data_set]
            prepared_batch_data.update(
                self.prepare_one_batch_context_and_label_data(one_batch_data, self.char_max_len, word2id,
                                                              word_embedded_dic,
                                                              char2id))
            yield_batch_data[data_set] = prepared_batch_data
        yield yield_batch_data
        test_data_generator = data_generator_dic[test_data_set]
        for one_batch_data in test_data_generator:
            prepared_batch_data = self.prepare_one_batch_meta_data(one_batch_data, user2id, country2id,
                                                                   client2id, session2id, format2id)
            word2id, word_embedded_dic, char2id = context_embedding_dic[test_data_set]
            prepared_batch_data.update(
                self.prepare_one_batch_context_and_label_data(one_batch_data, self.char_max_len, word2id,
                                                              word_embedded_dic,
                                                              char2id))
            yield_batch_data[test_data_set] = prepared_batch_data
            yield yield_batch_data


if __name__ == "__main__":
    g = MultiTaskDataLoader(is_training=True).generate_train_batch_data(3)
    for batch in g:
        print(batch['en_es']['client_input'])
