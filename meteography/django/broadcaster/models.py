import os.path
import threading

import matplotlib.pylab as plt

from django.db import models
from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.forms import CharField as CharFormField

from meteography.django.broadcaster.settings import WEBCAM_URL
from meteography.django.broadcaster.storage import webcam_fs


class CommaSeparatedIntegerFormField(CharFormField):
    def prepare_value(self, value):
        return ','.join(map(str, value)) if value else ''


class CommaSeparatedIntegerField(models.CommaSeparatedIntegerField):
    """
    Child of the django comma-separated field that behaves as one would expect
    (i.e. the python type is a sequence of int)
    """
    default_validators = []

    def from_db_value(self, value, expression, connection, context):
        if value is None:
            return value

        return map(int, value.split(','))

    def to_python(self, value):
        if len(value) and isinstance(value[0], int):
            return value

        if value is None:
            return value

        return map(int, value.split(','))

    def get_prep_value(self, value):
        return ','.join(map(str, value)) if value else ''

    def value_to_string(self, obj):
        value = self._get_val_from_obj(obj)
        return self.get_prep_value(value)

    def formfield(self, **kwargs):
        return CommaSeparatedIntegerFormField(**kwargs)


class Webcam(models.Model):
    webcam_id = models.SlugField(max_length=16, primary_key=True)
    name = models.CharField(max_length=32)

    def latest_prediction(self):
        predictions = Prediction.objects.filter(params__webcam=self)
        if predictions:
            latest = predictions.latest()
        else:
            latest = None
        return latest

    def store(self):
        webcam_fs.add_webcam(self.webcam_id)

    def post_delete(self):
        webcam_fs.delete_webcam(self.webcam_id)

    def __unicode__(self):
        return self.name


class Picture:
    def __init__(self, webcam, timestamp, fp):
        self.webcam = webcam
        self.timestamp = timestamp
        self.fp = fp

    def save(self):
        webcam_fs.add_picture(self.webcam, self.timestamp, self.fp)


class PredictionParams(models.Model):
    webcam = models.ForeignKey(Webcam)
    name = models.SlugField(max_length=16)
    intervals = CommaSeparatedIntegerField(max_length=128)

    def save(self, *args, **kwargs):
        """
        Create a set of examples in DB and in the dataset of the cam.
        The update of the dataset is done asynchronously as it may take time
        to read all the existing images of the cam.

        TODO: race conditions with image upload
        TODO: what if the dataset update fail? the DB is already updated

        Note:
        -----
        This will erase any set with the same name!
        """
        super(PredictionParams, self).save(*args, **kwargs)
        # FIXME saving without changing should not erase data (prevent in UI?)
        t = threading.Thread(target=webcam_fs.add_examples_set, args=[self])
        t.start()

    def post_delete(self):
        webcam_fs.delete_examples_set(self)

    def __unicode__(self):
        return '%s.%s: %s' % (
            self.webcam.webcam_id, self.name, str(self.intervals))


class Prediction(models.Model):
    params = models.ForeignKey(PredictionParams)
    comp_date = models.DateTimeField('computation date')
    path = models.CharField(max_length=100)

    class Meta:
        get_latest_by = 'comp_date'

    def url(self):
        return os.path.join(WEBCAM_URL, self.path)

    def minutes_target(self):
        return (self.params.intervals[-1] // 60)

    def save(self, *args, **kwargs):
        with webcam_fs.fs.open(self.path, 'w') as fp:
            plt.imsave(fp, self.sci_bytes)
        super(Prediction, self).save(*args, **kwargs)


@receiver(post_delete, sender=PredictionParams)
@receiver(post_delete, sender=Webcam)
def models_post_delete(sender, instance, **kwargs):
    instance.post_delete()
