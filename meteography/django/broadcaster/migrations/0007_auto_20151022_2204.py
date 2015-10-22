# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
import meteography.django.broadcaster.models


class Migration(migrations.Migration):

    dependencies = [
        ('broadcaster', '0006_auto_20151021_2244'),
    ]

    operations = [
        migrations.AlterField(
            model_name='predictionparams',
            name='intervals',
            field=meteography.django.broadcaster.models.CommaSeparatedIntegerField(max_length=128),
        ),
    ]
