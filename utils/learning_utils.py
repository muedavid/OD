import numpy as np
import tensorflow as tf
from datetime import datetime
import os

import utils.tools as tools


class LearningUtil:

    def __init__(self, bs, config):
        self.bs = bs
        self.cfg = tools.config_loader(os.path.join(config, 'training.yaml'))

    def get_lr(self, img_count):
        decay_step = np.ceil(img_count / self.bs) * self.cfg["EPOCHS"]
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(self.cfg['LR']['START'], decay_steps=decay_step,
                                                                    end_learning_rate=self.cfg['LR']['END'],
                                                                    power=self.cfg['LR']['POWER'])
        return lr_schedule

    def get_callbacks(self, img_count, paths):
        freq = int(np.ceil(img_count / self.bs) * self.cfg["CALLBACKS"]["CKPT_FREQ"]) + 1

        callbacks = [tf.keras.callbacks.ModelCheckpoint(
            filepath=paths["CKPT"] + "/ckpt-loss={val_loss:.2f}-epoch={epoch:.2f}-f1={val_f1:.4f}",
            save_weights_only=False, save_best_only=False, monitor="val_f1", verbose=1, save_freq='epoch', period=self.cfg["CALLBACKS"]["CKPT_FREQ"])]

        if self.cfg['CALLBACKS']['LOG_TB']:
            logdir = os.path.join(paths['TBLOGS'], datetime.now().strftime("%Y%m%d-%H%M%S"))
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1))

        return callbacks
