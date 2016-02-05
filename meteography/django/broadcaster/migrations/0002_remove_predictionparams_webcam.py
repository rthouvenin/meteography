# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('broadcaster', '0001_squashed_0005_featureset_extract_type'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='predictionparams',
            name='webcam',
        ),
    ]
