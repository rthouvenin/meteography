# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('broadcaster', '0005_auto_20151021_2218'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='prediction',
            name='img_file',
        ),
        migrations.AddField(
            model_name='prediction',
            name='path',
            field=models.CharField(default='', max_length=100),
            preserve_default=False,
        ),
    ]
