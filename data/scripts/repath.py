#!/usr/bin/env python

import argparse
from datetime import datetime
import os
import shutil

import tables

parser = argparse.ArgumentParser(
    description="Store a flat list of pictures into directories")
parser.add_argument('--count', type=int)
parser.add_argument('--format', default='%Y/%m/%d/%t.jpg')
parser.add_argument('--files')
parser.add_argument('--h5')
args = parser.parse_args()


def get_target_file(timestamp):
    file_date = datetime.utcfromtimestamp(int(timestamp))
    file_format = args.format.replace('%t', timestamp)
    target_file = file_date.strftime(file_format)
    return target_file


if args.files is not None:
    files = os.listdir(args.files)
    files.sort()

    if args.count is not None:
        files = files[:args.count]

    for filename in files:
        if not os.path.isdir(filename):
            timestamp, __ = os.path.splitext(filename)
            target_file = get_target_file(timestamp)
            target_path = os.path.join(args.files, target_file)
            dirname = os.path.dirname(target_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            shutil.copy(filename, target_path)

if args.h5 is not None:
    fp = tables.open_file(args.h5, 'a')
    rows = fp.root.images.imgset.iterrows()
    for r in rows:
        basename = os.path.basename(r['name'])
        timestamp = str(r['time'])
        target_file = get_target_file(timestamp)
        r['name'] = r['name'].replace(basename, target_file)
        r.update()
    fp.close()
