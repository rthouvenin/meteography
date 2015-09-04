#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to collect snapshots from a Wunderground webcam and save them on disk
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
    description="Collects snapshots from a Wunderground webcam.")
parser.add_argument('location', help="XX/City where XX is state or country code")
parser.add_argument('webcamid', help="The webcam identifier")
args = parser.parse_args();

CAMS_DIR = './webcams'
API_BASE_URL = 'http://api.wunderground.com/api/'
API_KEY = 'NO_KEY'
with open('WU_API_KEY') as f:
    API_KEY = f.read().strip()

logging.basicConfig(format='%(asctime)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_webcam_data(loc_query, webcam_name):
    """
    Queries the webcam API with loc_query, 
    and returns the json data for the webcam webcam_name
    """
    url = '%s/%s/webcams/q/%s.json' % (API_BASE_URL, API_KEY, loc_query)
    response = urllib2.urlopen(url)
    json_response = json.load(response)
    if 'webcams' in json_response:
        webcams = json_response['webcams']
        webcams = [w for w in webcams if w['camid'] == webcam_name]
        if webcams:
            return util.JsonProxy(webcams[0])
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

def query_and_retrieve(location, webcam_name, latest_snapshot):
    """
    Queries the webcam API, and retrieves a snapshot if a recent one is available
    """
    try:
        json_webcam = get_webcam_data(location, webcam_name)
        logger.debug('%s', json_webcam)
        has_recent, latest_snapshot = has_recent_snapshot(json_webcam, latest_snapshot)
        if has_recent:
            retrieve_snapshot(json_webcam)
    except Exception as e:
        if e.message:
            logger.error(e.message)
        else:
            logger.exception('Unknown error')
    return latest_snapshot
    

collect_duration = 0 #in hours, 0 for infinite collection
query_interval = 5 #in minutes
latest_snapshot = int(time.time())

if collect_duration:
    for i in range(collect_duration * (60 / query_interval)):
        latest_snapshot = query_and_retrieve(args.location, args.webcamid, latest_snapshot)
        time.sleep(60 * query_interval) #sleep in seconds
else:
    while True:
        latest_snapshot = query_and_retrieve(args.location, args.webcamid, latest_snapshot)
        time.sleep(60 * query_interval) #sleep in seconds
