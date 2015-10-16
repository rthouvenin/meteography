from django.db import models

from meteography.django.broadcaster.storage import WebcamStorage

webcam_fs = WebcamStorage()


class Webcam(models.Model):
    webcam_id = models.CharField(max_length=16, primary_key=True)
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
        webcam_fs.add_picture(self.webcam.webcam_id, self.timestamp, self.fp)


class Prediction(models.Model):
    webcam = models.ForeignKey(Webcam)
    comp_date = models.DateTimeField('computation date')
    target = models.DurationField()
