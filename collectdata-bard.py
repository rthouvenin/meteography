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

# Arguments parsing
class DateAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super(DateAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        """Convert arg string value into datetime"""
        if values:
            val = datetime.strptime(values, '%Y%m%d%H%M')
        else:
            val = datetime.today()
        setattr(namespace, self.dest, val)

parser = argparse.ArgumentParser(
    description="Collects webcam archives from meteobard.fr")
parser.add_argument('start', action=DateAction,
                    help="Start date, format yyyymmddhhmm")
parser.add_argument('end', nargs='?', action=DateAction,
                    help="End date, format yyyymmddhhmm, default: current date")
args = parser.parse_args();

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
    time.sleep(1) #Let's not be too harsh on the server