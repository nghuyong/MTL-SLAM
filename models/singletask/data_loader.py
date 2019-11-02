#!/usr/bin/env python
# encoding: utf-8
from models.base.data_loader import DataLoader


class SingleTaskDataLoader(DataLoader):
    def __init__(self, is_training):
        super().__init__(is_training)

    def generate_train_batch_data(self, one_data_set_batch_size, data_set):
        data_generator = self.batch_data_generator(self.get_data_path(data_set),
                                                   one_data_set_batch_size)
        word2id, word_embedded_dic, char2id = self.load_context_embedding(data_set)
        country2id, user2id, client2id, format2id, session2id = self.load_meta_embedding()
        while True:
            one_batch_data = data_generator.__next__()

            prepared_batch_data = self.prepare_one_batch_meta_data(one_batch_data, user2id, country2id,
                                                                   client2id, session2id, format2id)
            prepared_batch_data.update(
                self.prepare_one_batch_context_and_label_data(one_batch_data, self.char_max_len, word2id,
                                                                 word_embedded_dic, char2id))
            yield prepared_batch_data

    def generate_test_batch_data(self, one_data_set_batch_size, data_set):
        data_generator = self.batch_data_generator(self.get_data_path(data_set),
                                                   one_data_set_batch_size)
        word2id, word_embedded_dic, char2id = self.load_context_embedding(data_set)
        country2id, user2id, client2id, format2id, session2id = self.load_meta_embedding()
        for one_batch_data in data_generator:
            prepared_batch_data = self.prepare_one_batch_meta_data(one_batch_data, user2id, country2id,
                                                                   client2id, session2id, format2id)
            prepared_batch_data.update(
                self.prepare_one_batch_context_and_label_data(one_batch_data, self.char_max_len, word2id,
                                                                 word_embedded_dic, char2id))
            yield prepared_batch_data


if __name__ == "__main__":
    g = SingleTaskDataLoader(is_training=True).generate_test_batch_data(128, 'en_es')
    for batch in g:
        print(batch['word_embedded'])
