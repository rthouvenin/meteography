# -*- coding: utf-8 -*-
import os.path

from django.conf import settings

PREDICTIONS_DIR = getattr(settings, PREDICTIONS_DIR, 
                          os.path.join(settings.BASE_DIR, 'temp', 'predict'))