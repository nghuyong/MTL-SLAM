#!/usr/bin/env python
# encoding: utf-8
from models.base.run import BaseRun
from models.multitask.train import MultitaskTrainer
from models.multitask.data_loader import MultiTaskDataLoader
from models.multitask.model import MultiTaskModel
from models.multitask.config import MultiTaskConfig


class MultiTaskRun(BaseRun):
    def trainer_init(self):
        return MultitaskTrainer(MultiTaskDataLoader)

    def model_init(self):
        return MultiTaskModel(MultiTaskConfig(), self.ARGS.model_id)


if __name__ == "__main__":
    MultiTaskRun().run()
