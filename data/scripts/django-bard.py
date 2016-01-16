#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
import os.path
import time

import requests

import util

parser = argparse.ArgumentParser(
    description="Uploads pictures from a bard archive to brodcaster API")
parser.add_argument('webcam')
parser.add_argument('day', action=util.DateAction)
parser.add_argument('--host', default='localhost')
args = parser.parse_args()


url_pattern = 'http://%s/webcams/%s/pictures/%d'
campath = os.path.join('..', 'webcams', args.webcam)

epoch = datetime.fromtimestamp(0)
day = datetime(args.day.year, args.day.month, args.day.day)
daypath = os.path.join(campath, str(day.year), "%02d" % day.month, "%02d" % day.day)

hours = os.listdir(daypath)
hours.sort()
for h in hours:
    filenames = os.listdir(os.path.join(daypath, h))
    filenames.sort()
    for filename in filenames:
        with open(os.path.join(daypath, h, filename)) as fp:
            m = int(os.path.splitext(filename)[0][-2:])
            dt = datetime(day.year, day.month, day.day, int(h), int(m))
            timestamp = (dt - epoch).total_seconds()

            url = url_pattern % (args.host, args.webcam, timestamp)
            img_data = fp.read()

            start_time = time.time()
            resp = requests.put(url, data=img_data)
            elapsed = time.time() - start_time

            print("%s/%s: %d [%s] in %.3fs"
                  % (h, filename, resp.status_code, resp.reason, elapsed))
