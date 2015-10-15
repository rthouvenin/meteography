# -*- coding: utf-8 -*-
import os.path

from django.conf import settings

WEBCAM_DIR = getattr(settings, 'WEBCAM_DIR',
                     os.path.join(settings.MEDIA_ROOT, 'webcams'))
