#!/usr/bin/env python
# encoding: utf-8
from models.multitask.config import MultiTaskConfig


class SingleTaskConfig(MultiTaskConfig):
    def __init__(self, data_set):
        super().__init__()
        self.data_set = data_set
