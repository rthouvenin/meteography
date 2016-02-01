# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
import meteography.django.broadcaster.models


class Migration(migrations.Migration):

    replaces = [(b'broadcaster', '0001_initial'), (b'broadcaster', '0002_webcam_compressed'), (b'broadcaster', '0003_prediction_error'), (b'broadcaster', '0004_auto_20160201_1510'), (b'broadcaster', '0005_featureset_extract_type')]

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Prediction',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('comp_date', models.DateTimeField(verbose_name=b'computation date')),
                ('path', models.CharField(max_length=100)),
            ],
            options={
                'get_latest_by': 'comp_date',
            },
        ),
        migrations.CreateModel(
            name='PredictionParams',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.SlugField(max_length=16)),
                ('intervals', meteography.django.broadcaster.models.CommaSeparatedIntegerField(max_length=128)),
            ],
        ),
        migrations.CreateModel(
            name='Webcam',
            fields=[
                ('webcam_id', models.SlugField(max_length=16, serialize=False, primary_key=True)),
                ('name', models.CharField(max_length=32)),
            ],
        ),
        migrations.AddField(
            model_name='predictionparams',
            name='webcam',
            field=models.ForeignKey(to='broadcaster.Webcam'),
        ),
        migrations.AddField(
            model_name='prediction',
            name='params',
            field=models.ForeignKey(to='broadcaster.PredictionParams'),
        ),
        migrations.AddField(
            model_name='prediction',
            name='error',
            field=models.FloatField(null=True),
        ),
        migrations.CreateModel(
            name='FeatureSet',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.SlugField(max_length=16)),
                ('webcam', models.ForeignKey(to='broadcaster.Webcam')),
                ('extract_type', models.CharField(default='raw', max_length=16, choices=[(b'raw', b'Raw'), (b'pca', b'PCA')])),
            ],
        ),
        migrations.AddField(
            model_name='predictionparams',
            name='features',
            field=models.ForeignKey(default=None, to='broadcaster.FeatureSet'),
            preserve_default=False,
        ),
    ]
