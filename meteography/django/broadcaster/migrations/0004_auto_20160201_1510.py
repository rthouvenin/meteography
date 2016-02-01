# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('broadcaster', '0003_prediction_error'),
    ]

    operations = [
        migrations.CreateModel(
            name='FeatureSet',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.SlugField(max_length=16)),
            ],
        ),
        migrations.RemoveField(
            model_name='webcam',
            name='compressed',
        ),
        migrations.AddField(
            model_name='featureset',
            name='webcam',
            field=models.ForeignKey(to='broadcaster.Webcam'),
        ),
        migrations.AddField(
            model_name='predictionparams',
            name='features',
            field=models.ForeignKey(default=None, to='broadcaster.FeatureSet'),
            preserve_default=False,
        ),
    ]
