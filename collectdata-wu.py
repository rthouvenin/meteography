#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to collect snapshots from Wunderground webcams and save them on disk
with the timestamp as filename
TODO Better options for API key, read latest snapshot date from disk, mkdir

@author: romain
"""

import argparse
import json
import logging
import os.path
import time
import urllib2

import util

parser = argparse.ArgumentParser(
    description="Collects snapshots from Wunderground webcam(s).")
parser.add_argument('location', help="XX/City with XX = state or country code")
parser.add_argument('webcamids', nargs='+', help="The webcam identifier(s)")
args = parser.parse_args()

CAMS_DIR = './webcams'
API_BASE_URL = 'http://api.wunderground.com/api/'
API_KEY = 'NO_KEY'
with open('WU_API_KEY') as f:
    API_KEY = f.read().strip()

logging.basicConfig(format='%(asctime)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_webcam_data(loc_query, webcam_names):
    """
    Queries the webcam API with loc_query,
    and returns the json data for the webcams of webcam_names in a dictionary
    indexed by the webcam names
    """
    url = '%s/%s/webcams/q/%s.json' % (API_BASE_URL, API_KEY, loc_query)
    response = urllib2.urlopen(url)
    json_response = json.load(response)
    if 'webcams' in json_response:
        webcams = json_response['webcams']
        webcams = {w['camid']: util.JsonProxy(w) for w in webcams
                   if w['camid'] in webcam_names}
        if webcams:
            return webcams
        else:
            raise Exception('Webcam not found.')
    else:
        if 'response' in json_response and 'error' in json_response['response']:
            err = json_response['response']['error']
            msg = 'Error while retrieving webcam data: %s' % err['description']
        else:
            msg = 'Unknown error while retrieving webcam data'
        raise Exception(msg)


def image_name(webcam):
    """
    Generates the name of the file where to store the webcam snapshot
    """
    return os.path.join(CAMS_DIR, webcam.camid, webcam.updated_epoch + '.jpg')


def has_recent_snapshot(webcam, latest_snapshot):
    """
    Returns a tuple with a boolean telling whether the webcam image was updated
    after latest_snapshot, and the latest of the two dates
    """
    updated = int(webcam.updated_epoch)
    return updated > latest_snapshot, max(updated, latest_snapshot)


def retrieve_snapshot(webcam):
    """
    Retrieves a snapshot from webcam_url and stores it.
    """
    imgrequest = urllib2.urlopen(webcam.CURRENTIMAGEURL)
    filename = image_name(webcam)
    with open(filename, 'wb') as f:
        try:
            f.write(imgrequest.read())
            logger.info('Wrote %s', filename)
        except:
            logger.exception('Error while retrieving snapshot %s', filename)


def retrieve(json_webcam, latest_snapshot):
    """
    Retrieves a snapshot if a recent one is available
    """
    logger.debug('%s', json_webcam)
    has_recent, latest_snapshot = has_recent_snapshot(json_webcam, latest_snapshot)
    if has_recent:
        retrieve_snapshot(json_webcam)
    return latest_snapshot


def query_and_retrieve(location, webcamids, latests):
    """
    Queries from the API the webcam data and retrieves a snapshot from each
    of them if available
    """
    try:
        webcams = get_webcam_data(location, webcamids)
        for w, wid in enumerate(webcams):
            latests[w] = retrieve(webcams[wid], latests[w])
    except Exception as e:
        if e.message:
            logger.error(e.message)
        else:
            logger.exception('Unknown error')

#Creates webcam directories if required
for wid in args.webcamids:
    wdir = os.path.join(CAMS_DIR, wid)
    if not os.path.exists(wdir):
        os.mkdir(wdir)

#Starts retrieval
collect_duration = 0  # in hours, falsy value for infinite collection
query_interval = 5  # in minutes
t = int(time.time())
latests = [t] * len(args.webcamids)

if collect_duration:
    for i in range(collect_duration * (60 / query_interval)):
        query_and_retrieve(args.location, args.webcamids, latests)
        time.sleep(60 * query_interval)  # sleep in seconds
else:
    while True:
        query_and_retrieve(args.location, args.webcamids, latests)
        time.sleep(60 * query_interval)  # sleep in seconds
