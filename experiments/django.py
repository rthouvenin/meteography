#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path

import requests

filepath = os.path.join('..', 'data', 'rednight', '10.png')
url_pattern = 'http://localhost:8000/webcams/%s/pictures/%s'
webcam = 'sfo'

with open(filepath) as f:
    filename = os.path.basename(filepath)
    url = url_pattern % (webcam, filename)
    resp = requests.put(url, data=f.read())
    print("%s: %d [%s]" % (filename, resp.status_code, resp.reason))
