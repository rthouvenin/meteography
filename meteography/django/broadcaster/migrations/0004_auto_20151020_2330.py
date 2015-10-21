# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('broadcaster', '0003_auto_20151018_2235'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='prediction',
            options={'get_latest_by': 'comp_date'},
        ),
    ]
