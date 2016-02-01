# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('broadcaster', '0004_auto_20160201_1510'),
    ]

    operations = [
        migrations.AddField(
            model_name='featureset',
            name='extract_type',
            field=models.CharField(default='raw', max_length=16, choices=[(b'raw', b'Raw'), (b'pca', b'PCA')]),
            preserve_default=False,
        ),
    ]
