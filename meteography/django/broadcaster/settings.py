# -*- coding: utf-8 -*-
import os.path

from django.conf import settings

WEBCAM_DIR = getattr(settings, 'WEBCAM_DIR',
                     os.path.join(settings.MEDIA_ROOT, 'webcams'))

WEBCAM_SIZE = getattr(settings, 'WEBCAM_SIZE', (80, 60))

SET_NAME = 'online'
