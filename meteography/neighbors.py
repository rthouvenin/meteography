# -*- coding: utf-8 -*-
"""
Wrapper around sklearn k-neighbors estimators that can work in batches on
pytables arrays (or other disk-backed arrays that support slicing)
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors as SKNN


class NearestNeighbors:
    BATCH_SIZE = 20 * 1024 * 1024

    def __init__(self):
        self.sknn = SKNN(1, algorithm='brute')

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.batch_len = max(1, self.BATCH_SIZE // X.shape[1])
        self.nb_batch = 0
        self.batch = None
        if len(X) > 0:
            self._reset_nb_batch()

    def _reset_nb_batch(self):
        old = self.nb_batch
        self.nb_batch = len(self.X) // self.batch_len
        if len(self.X) % self.batch_len:
            self.nb_batch += 1

        oldincr = (old > 1)
        incr = (self.nb_batch > 1)
        if self.batch is None or oldincr != incr:
            self.batch = np.empty((self.batch_len+incr, self.X.shape[1]),
                                  np.float32)  # FIXME do not hardcode
        return self.nb_batch

    def _get_batch(self, b, extra_row):
        start = b * self.batch_len
        end = min(start+self.batch_len, len(self.X))
        actual_len = end - start
        self.batch[:actual_len] = self.X[start:end]
        has_extra = 0
        if extra_row is not None:
            has_extra = 1
            self.batch[actual_len] = self.X[extra_row]

        if actual_len+has_extra == self.batch.shape[0]:
            return self.batch
        else:
            return self.batch[:actual_len+has_extra]

    def predict(self, input_row):
        self._reset_nb_batch()
        nearest = None
        for b in range(self.nb_batch):
            batch = self._get_batch(b, nearest)
            self.sknn.fit(batch)
            i_batch = self.sknn.kneighbors([input_row], return_distance=False)
            i_batch = i_batch[0][0]
            if i_batch != (batch.shape[0]-1) or b == 0:
                nearest = b * self.batch_len + i_batch
        return self.y[nearest]
