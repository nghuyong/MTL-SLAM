#!/usr/bin/env python
# encoding: utf-8
from models.base.run import BaseRun
from models.singletask.config import SingleTaskConfig
from models.singletask.model import SingleTaskModel
from models.singletask.train import SingleTaskTrainer
from models.singletask.data_loader import SingleTaskDataLoader


class SingleTaskRun(BaseRun):
    def trainer_init(self):
        return SingleTaskTrainer(SingleTaskDataLoader)

    def model_init(self):
        return SingleTaskModel(SingleTaskConfig(self.ARGS.data_set), self.ARGS.model_id)


if __name__ == "__main__":
    SingleTaskRun().run()
