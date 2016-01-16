#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os.path
import time

import requests

#import util

parser = argparse.ArgumentParser(
    description="Uploads webcam pictures to brodcaster API")
parser.add_argument('webcam')
parser.add_argument('--host', default='localhost:8000')
parser.add_argument('--count', type=int)
parser.add_argument('--from-index', type=int)
#parser.add_argument('--from-date', action=util.DateAction)
args = parser.parse_args()


url_pattern = 'http://%s/webcams/%s/pictures/%s'
campath = os.path.join('..', 'webcams', args.webcam)
camfiles = os.listdir(campath)
camfiles.sort()

if args.from_index:
    camfiles = camfiles[args.from_index:]
if args.count:
    camfiles = camfiles[:args.count]

for camfile in camfiles:
    filepath = os.path.join(campath, camfile)
    with open(filepath) as f:
        timestamp, __ = os.path.splitext(camfile)
        url = url_pattern % (args.host, args.webcam, timestamp)
        img_data = f.read()

        start_time = time.time()
        resp = requests.put(url, data=img_data)
        elapsed = time.time() - start_time

        print("%s: %d [%s] in %.3fs"
              % (camfile, resp.status_code, resp.reason, elapsed))
        if resp.status_code != 204:
            oops_name = 'oops-' + timestamp + '.html'
            oops_path = os.path.join('..', 'temp', oops_name)
            with open(oops_path, 'w') as oopsf:
                oopsf.write(resp.text)
