# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
import django.core.files.storage


class Migration(migrations.Migration):

    dependencies = [
        ('broadcaster', '0004_auto_20151020_2330'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='prediction',
            name='path',
        ),
        migrations.AddField(
            model_name='prediction',
            name='img_file',
            field=models.FileField(default='', storage=django.core.files.storage.FileSystemStorage(b'/home/romain/prog/meteography/website/temp/webcams'), upload_to=b''),
            preserve_default=False,
        ),
    ]
