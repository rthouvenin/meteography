#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
from datetime import timedelta
import os.path
import time

import requests

parser = argparse.ArgumentParser(
    description="Uploads pictures from a bard archive to brodcaster API")
parser.add_argument('webcam')
parser.add_argument('fromday')
parser.add_argument('today')
parser.add_argument('--host', default='localhost:8000')
args = parser.parse_args()


url_pattern = 'http://%s/webcams/%s/pictures/%d'
campath = os.path.join('..', 'webcams', args.webcam)

epoch = datetime.fromtimestamp(0)
from_day = datetime.strptime(args.fromday, '%Y%m%d')
to_day = datetime.strptime(args.today, '%Y%m%d')
nb_days = (to_day - from_day).days

for d in range(nb_days+1):
    day = from_day + timedelta(d)
    daypath = os.path.join(campath, day.strftime("%Y/%m/%d"))

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
