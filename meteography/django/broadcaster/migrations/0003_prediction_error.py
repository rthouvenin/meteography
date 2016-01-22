# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('broadcaster', '0002_webcam_compressed'),
    ]

    operations = [
        migrations.AddField(
            model_name='prediction',
            name='error',
            field=models.FloatField(null=True),
        ),
    ]
