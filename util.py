# -*- coding: utf-8 -*-
"""
Utilities

@author: romain
"""

class JsonProxy:
    """
    Class to be used as a proxy to the dictionaries returned by json.load(),
    so as to be able to retrieve values with attribute expression.
    For example, be able to write resp.errors instead of resp['errors']
    """
    def __init__(self, d):
        """Constructor with the object to wrap"""
        self.source = d
    
    def __getattr__(self, name):
        """Gets self.source[name] and wraps it if needed"""
        if name in self.source:
            val = self.source[name]
            if isinstance(val, (dict, list)):
                return JsonProxy(val)
            else:
                return val
        raise AttributeError(name)
    
    def __getitem__(self, key):
        """Gets self.source[key] and wraps it if needed"""
        val = self.source.__getitem__(key)
        if isinstance(val, (dict, list)):
            return JsonProxy(val)
        else:
            return val
    
    def __repr__(self):
        """Delegates to source"""
        return self.source.__repr__()