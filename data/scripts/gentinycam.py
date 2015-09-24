#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path

import PIL as pil

source_path = os.path.join('..', 'RBWeatherCamCAM1')
dest_path = os.path.join('..', 'tinycam')

source_files = os.listdir(source_path)
for f in source_files:
    img = pil.Image.open(os.path.join(source_path, f))
    thumb = img.resize((117, 80), pil.Image.ANTIALIAS)
    thumb.save(os.path.join(dest_path, f))
