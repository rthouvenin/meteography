# -*- coding: utf-8 -*-
import os.path

from django.conf import settings

WEBCAM_DIR = getattr(settings, 'WEBCAM_DIR', 'webcams')
WEBCAM_ROOT = getattr(settings, 'WEBCAM_ROOT',
                      os.path.join(settings.MEDIA_ROOT, WEBCAM_DIR))
WEBCAM_URL = getattr(settings, 'WEBCAM_URL',
                     os.path.join(settings.MEDIA_URL, WEBCAM_DIR))

WEBCAM_SIZE = getattr(settings, 'WEBCAM_SIZE', (80, 60))
