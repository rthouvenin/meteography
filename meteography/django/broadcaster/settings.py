# -*- coding: utf-8 -*-
import os.path

from django.conf import settings


WEBCAM_URL = getattr(settings, 'WEBCAM_URL', 'webcams')
WEBCAM_ROOT = getattr(settings, 'WEBCAM_ROOT',
                      os.path.join(settings.BASE_DIR, 'cams'))

WEBCAM_STATIC_URL = getattr(settings, 'WEBCAM_STATIC_URL', 'cams')

WEBCAM_SIZE = getattr(settings, 'WEBCAM_SIZE', (80, 60))
DEFAULT_FEATURES_SIZE = getattr(settings, 'DEFAULT_FEATURES_SIZE', (40, 30))

# %t means timestamp as an integer
PICTURE_PATH = getattr(settings, 'PICTURE_PATH', '%Y/%m/%d/%t.jpg')
