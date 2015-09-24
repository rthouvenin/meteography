#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to collect snapshots from Wunderground webcams.
Does not use the API, but instead read the images from a list of URLs,
and assumes the http header indicates when the picture was taken.
"""

import logging
import os.path
import rfc822
import time
import urllib2

URL_FILE = 'urls.txt'
WEBCAM_DIR = os.path.join('..', 'webcams')
LAST_UPDATE_HEADER = 'Last-Modified'
QUERY_INTERVAL = 5  # in minutes

logging.basicConfig(format='%(asctime)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def parse_url(url):
    """Extract webcam data from its URL"""
    url_parts = url.split('/')
    webcam_name = url_parts[-3] + 'CAM' + url_parts[-2]
    file_ext = url[-5:-1]
    last_update = 0.
    return {
        'url': url[:-1],  # Skip end of line
        'name': webcam_name,
        'imgpath': os.path.join(WEBCAM_DIR, webcam_name, '%d' + file_ext),
        'last_update': last_update
        }

urls = open(URL_FILE, 'r').readlines()
webcams = map(parse_url, urls)

#Create missing directories
for cam in webcams:
    folder = os.path.dirname(cam['imgpath'])
    if not os.path.exists(folder):
        os.mkdir(folder)

#Start retrieving
while True:
    for url, webcam in zip(urls, webcams):
        try:
            imgrequest = urllib2.urlopen(url)
            headers = imgrequest.info()
            if LAST_UPDATE_HEADER in headers:
                last_update = headers[LAST_UPDATE_HEADER]
                last_update = time.mktime(rfc822.parsedate(last_update))
                if last_update > webcam['last_update']:
                    webcam['last_update'] = last_update
                    filepath = webcam['imgpath'] % int(last_update)
                    with open(filepath, 'wb') as f:
                        f.write(imgrequest.read())
                        logger.info('Wrote %s', filepath)
        except:
            logger.exception('Error while retrieving %s', url)
    time.sleep(60 * QUERY_INTERVAL)
