# Inspired from http://stackoverflow.com/a/8759188/817766

from threading import currentThread

from meteography.dataset import DataSet

_request_cache = {}
_installed_middleware = False


def get_dataset_cache():
    assert _installed_middleware, 'RequestCacheMiddleware not loaded'
    return _request_cache[currentThread()]


class DataSetCache(object):
    def __init__(self):
        self._cache = dict()

    def __getitem__(self, key):
        dataset = self._cache.get(key)
        if dataset is None:
            dataset = DataSet.open(key)
            self._cache[key] = dataset
        return dataset

    def clear(self):
        for dataset in self._cache.values():
            dataset.close()
        self._cache.clear()


class RequestCacheMiddleware(object):
    def __init__(self):
        global _installed_middleware
        _installed_middleware = True

    def process_request(self, request):
        _request_cache[currentThread()] = DataSetCache()

    def clean_cache(self):
        t = currentThread()
        cache = _request_cache.get(t)
        if cache is not None:
            cache.clear()
            del _request_cache[t]

    def process_response(self, request, response):
        self.clean_cache()
        return response

    def process_exception(self, request, exception):
        self.clean_cache()
