import matplotlib.pylab as plt

from django.db import models

from meteography.django.broadcaster.storage import webcam_fs


class Webcam(models.Model):
    webcam_id = models.SlugField(max_length=16, primary_key=True)
    name = models.CharField(max_length=32)

    def store(self):
        webcam_fs.add_webcam(self.webcam_id)

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
    intervals = models.CommaSeparatedIntegerField(max_length=128)

    def save(self, *args, **kwargs):
        """
        Create an set of examples in the dataset of the cam before saving in DB

        Note:
        -----
        This will erase any set with the same name!
        """
        # converts csv string into list of ints, and back into string
        if len(self.intervals):
            self.intervals = map(int, self.intervals.split(','))
        # FIXME saving without changing should not erase data...
        webcam_fs.add_examples_set(self)
        self.intervals = ','.join(map(str, self.intervals))

        super(PredictionParams, self).save(*args, **kwargs)

    def __unicode__(self):
        return '%s.%s: [%s]' % (
            self.webcam.webcam_id, self.name, str(self.intervals))


class Prediction(models.Model):
    params = models.ForeignKey(PredictionParams)
    comp_date = models.DateTimeField('computation date')
    path = models.CharField(max_length=100)

    def save(self, *args, **kwargs):
        plt.imsave(self.path, self.sci_bytes)
        super(Prediction, self).save(*args, **kwargs)
