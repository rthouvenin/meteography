# -*- coding: utf-8 -*-
"""
Wrapper around sklearn k-neighbors estimators that can work in batches on
pytables arrays (or other disk-backed arrays that support slicing)
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors as SKNN


class NearestNeighbors:
    BATCH_SIZE = 10 * 1024 * 1024

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
        self.batch[:end-start] = self.X[start:end]
        if self.nb_batch > 1:
            #Copy the extra_row only if needed to save some cycles
            if extra_row is None:
                self.extra_row = 0
                self.batch[-1] = self.X[self.extra_row]
            elif extra_row != self.extra_row:
                self.extra_row = extra_row
                self.batch[-1] = self.X[self.extra_row]
        return self.batch

    def predict(self, input_row):
        self._reset_nb_batch()
        nearest = None
        for b in range(self.nb_batch):
            batch = self._get_batch(b, nearest)
            self.sknn.fit(batch)
            nearest = self.sknn.kneighbors([input_row], return_distance=False)
            nearest = nearest[0][0]
            if self.nb_batch > 1 and nearest != self.batch_len:
                nearest = (b-1) * self.batch_len + nearest
        return self.y[nearest]
