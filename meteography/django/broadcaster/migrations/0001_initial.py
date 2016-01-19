# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
import meteography.django.broadcaster.models


class Migration(migrations.Migration):

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
    ]
