# -*- coding: utf-8 -*-

import random

import pytest

from meteography.dataset import ImageSet

class TestImageSet:
    @staticmethod 
    @pytest.fixture(scope='class')
    def imageset():
        images = [{'time': 4200}, {'time': 420}, {'time': 4242}, {'time': 42}]
        return ImageSet(images)
    
    def helper_closest_match(self, imageset, start, interval, expected):
        closest = imageset.find_closest(start, interval)
        assert(closest is not None)
        assert(closest['time'] == expected)
    
    def helper_closest_nomatch(self, imageset, start, interval):
        closest = imageset.find_closest(start, interval)
        assert(closest is None)
        
    def test_init(self, imageset):
        "The first element should have the smallest time"
        assert(imageset.time(0) == 42)

    def test_closest_last_inrange(self, imageset):
        "The target is past the last element, the last element is acceptable"
        self.helper_closest_match(imageset, 4210, 50, 4242)
    
    def test_closest_last_outrange(self, imageset):
        "The target is past the last element, the last element is not acceptable"
        self.helper_closest_nomatch(imageset, 4250, 50)
    
    def test_closest_exactmatch(self, imageset):
        "The target exists in the set"
        self.helper_closest_match(imageset, 400, 20, 420)
    
    def test_closest_matchleft_inrange(self, imageset):
        "The closest match is lower than the target and acceptable"
        self.helper_closest_match(imageset, 400, 40, 420)
    
    def test_closest_matchleft_outrange(self, imageset):
        "The closest match is lower than the target but not acceptable"
        self.helper_closest_nomatch(imageset, 420, 40)
        
    def test_closest_matchright_inrange(self, imageset):
        "The closest match is greater than the target and acceptable"
        self.helper_closest_match(imageset, 4100, 150, 4242)
        
    def test_closest_matchright_outrange(self, imageset):
        "The closest match is greater than the target but not acceptable"
        self.helper_closest_nomatch(imageset, 4000, 50)
    
    @staticmethod 
    @pytest.fixture(scope='class')
    def bigimageset():
        random.seed(42) #Reproducible tests
        def jitter(x):
            return x + random.randint(-2, 2)
        images = [{'time': jitter(t)} for t in range(1000, 2000, 10)]
        return ImageSet(images)
    
    def test_datapoints(self, bigimageset):
        "The list of data points should have the expected dimensions"
        data_points = bigimageset.find_datapoints(5, 10, 20)
        # (max - future_time - (hist_len - 1)*interval - min) / step + 1
        # = (2000 - 20 - 4*10 - 1000) / 10 + 1 = 95
        assert(len(data_points) == 95)
        X, y = random.choice(data_points)
        assert(len(X) == 5)
        intervals = [X[i+1]['time'] - X[i]['time'] for i in range(len(X)-1)]
        assert(min(intervals) <= 2*10)
        assert(y['time'] - X[-1]['time'] <= 2*20)