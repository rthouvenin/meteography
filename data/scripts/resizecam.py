#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path
import sys

import PIL.Image


def compute_shape(src_shape, dest_shape):
    if all(dest_shape):
        return dest_shape
    elif dest_shape[0]:
        return dest_shape[0], int(src_shape[1] * dest_shape[0] / src_shape[0])
    elif dest_shape[1]:
        return dest_shape[1], int(src_shape[0] * dest_shape[1] / src_shape[1])
    else:
        raise ValueError("empty shape")


def scan_dir(source, dest, shape):
    if not os.path.exists(dest):
        os.mkdir(dest)
    for f in os.listdir(source):
        fullname = os.path.join(source, f)
        if os.path.isdir(fullname):
            scan_dir(fullname, os.path.join(dest, f), shape)
        else:
            img = PIL.Image.open(fullname)
            dest_shape = compute_shape(img.size, shape)
            thumb = img.resize(dest_shape, PIL.Image.ANTIALIAS)
            thumb.save(os.path.join(dest, f))


source_path = sys.argv[1]
dest_path = sys.argv[2]
strshape = sys.argv[3].split('x')
shape = tuple(int(s) if s else None for s in strshape)
scan_dir(source_path, dest_path, shape)
