from keras.callbacks import Callback
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt


##########################
########## SGDR ##########
##########################


class SGDR(Callback):
    '''Cosine annealing learning rate scheduler with periodic restarts.
    # Usage
        ```python
            sgdr = SGDR(min_lr=1e-5,
                        max_lr=1e-2,
                        step_size=np.ceil(X_train.shape[0]/batch_size),
                        lr_decay=0.9,
                        mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[sgdr])
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        step_size: Size of the initial cycle. Calculated as `np.ceil(X_train.shape[0]/batch_size)`. 
        lr_decay: Reduce the max_lr after the completion of each cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    # References
        Original paper: http://arxiv.org/abs/1608.03983
    '''
    def __init__(self, min_lr, max_lr, step_size, lr_decay=1, mult_factor=2):

        super(SGDR, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.iterations = 0.

        self.step_size = step_size
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.step_size)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        
        if self.batch_since_restart + 1 == self.step_size:
            self.batch_since_restart = 0
            self.step_size = np.ceil(self.step_size * self.mult_factor)
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()
            
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)