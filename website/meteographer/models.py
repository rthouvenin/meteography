from django.db import models


class Webcam(models.Model):
    webcam_id = models.CharField(max_length=16, primary_key=True)
    name = models.CharField(max_length=32)
    
    def __unicode__(self):
        return self.name


class Prediction(models.Model):
    webcam = models.ForeignKey(Webcam)
    comp_date = models.DateTimeField('computation date')
    pred_date = models.DateTimeField('prediction date')
    
