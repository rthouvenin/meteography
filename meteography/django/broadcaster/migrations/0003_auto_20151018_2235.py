# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
import datetime
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('broadcaster', '0002_remove_prediction_webcam'),
    ]

    operations = [
        migrations.RenameField(
            model_name='prediction',
            old_name='timestamp',
            new_name='comp_date',
        ),
        migrations.AddField(
            model_name='prediction',
            name='path',
            field=models.CharField(default=datetime.datetime(2015, 10, 18, 20, 35, 36, 583080, tzinfo=utc), max_length=100),
            preserve_default=False,
        ),
    ]
