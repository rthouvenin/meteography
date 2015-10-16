#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path
import time

import requests

filepath = os.path.join('..', 'data', 'rednight', '10.png')
url_pattern = 'http://localhost:8000/webcams/%s/pictures/%s'
webcam = 'sfo'

with open(filepath) as f:
    filename, ext = os.path.splitext(os.path.basename(filepath))
    url = url_pattern % (webcam, filename)
    img_data = f.read()
    start_time = time.time()
    resp = requests.put(url, data=img_data)
    elapsed = time.time() - start_time
    print("%s: %d [%s] in %.3fs"
          % (filename, resp.status_code, resp.reason, elapsed))
