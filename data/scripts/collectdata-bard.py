#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to collect webcam archives from meteobard.fr
"""

import argparse
from datetime import datetime
from datetime import timedelta
import logging
import os.path
import time
import urllib2

import util

parser = argparse.ArgumentParser(
    description="Collects webcam archives from meteobard.fr")
parser.add_argument('start', action=util.DateAction,
                    help="Start date, format yyyymmddhhmm")
parser.add_argument('end', nargs='?', action=util.DateAction,
                    help="End date, format yyyymmddhhmm, default: current date")
args = parser.parse_args()

#Logging
logging.basicConfig(format='%(asctime)s: %(message)s')

#Constants
IMG_URL = "http://meteobard.fr/Pixxerbase/q1602/%Y/%m/%d/%H/m12-vga-%M.jpg"
IMG_PATH = 'webcams/bard'
IMG_FREQUENCY = timedelta(minutes=5)

#Actual script
current_time = args.start
while current_time <= args.end:
    url = current_time.strftime(IMG_URL)
    timestamp = (current_time - datetime.fromtimestamp(0)).total_seconds()
    filename = os.path.join(IMG_PATH, '%s.jpg' % int(timestamp))
    try:
        response = urllib2.urlopen(url)
        with open(filename, 'wb') as f:
            f.write(response.read())
    except:
        logging.error("Could not retrieve picture taken at %s." % current_time)
    current_time += IMG_FREQUENCY
    time.sleep(1)  # Let's not be too harsh on the server
