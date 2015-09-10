# -*- coding: utf-8 -*-
"""
One-time-use script to generate the dataset for bigrednight 
from the rednight pictures.
"""

import os.path
import random
import shutil

import PIL as pil

source_path = '../rednight/data'
dest_path = 'data'

source_files = os.listdir(source_path)
file_ids = [int(f[:-4]) for f in source_files]
file_ids.sort()

#duplicate the files 3 times
os.mkdir(dest_path)
for i in range(1,4):
    for f in source_files:
        sf = os.path.join(source_path, f)
        df = os.path.join(dest_path, '%d-%s' % (i,f))
        shutil.copy(sf, df)

#introduce averages with randomized weights between the existing ones
#and double the dimensions of each image
for i in range(1,4):
    for j,fid in enumerate(file_ids):
        f1 = os.path.join(dest_path, '%d-%d.png' % (i, fid))
        nextfid = file_ids[(j+1) % len(file_ids)]
        f2 = os.path.join(dest_path, '%d-%d.png' % (i, nextfid))
        i1 = pil.Image.open(f1)
        i2 = pil.Image.open(f2)
        w,h = i1.size
        nb_avg = random.choice([1,2,3])
        for k in range(nb_avg):
            coef = float(k+1)/(nb_avg+1)
            f3id = '%01d%02d%d' % (i-1, fid-1, int(coef*10))
            f3 = os.path.join(dest_path, '%s.png' % f3id)
            i3 = pil.Image.blend(i1, i2, coef)
            i3.resize((w*2, h*2), pil.Image.ANTIALIAS).save(f3)
        if j != 0: #will be needed for the last file
            os.remove(f1)
        f1 = os.path.join(dest_path, '%01d%02d0.png' % (i-1, fid-1))
        i1.resize((w*2, h*2), pil.Image.ANTIALIAS).save(f1)
            
            
    