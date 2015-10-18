# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Prediction',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('timestamp', models.DateTimeField(verbose_name=b'computation date')),
            ],
        ),
        migrations.CreateModel(
            name='PredictionParams',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.SlugField(max_length=16)),
                ('intervals', models.CommaSeparatedIntegerField(max_length=128)),
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
            name='webcam',
            field=models.ForeignKey(to='broadcaster.Webcam'),
        ),
    ]
