# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('broadcaster', '0002_remove_predictionparams_webcam'),
    ]

    operations = [
        migrations.AlterField(
            model_name='featureset',
            name='extract_type',
            field=models.CharField(max_length=16, choices=[(b'raw', b'Raw'), (b'pca', b'PCA'), (b'rbm', b'RBM')]),
        ),
    ]
