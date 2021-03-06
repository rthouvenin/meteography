from datetime import timedelta
import time

from django.db import models
from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.forms import CharField as CharFormField

from meteography.django.broadcaster.settings import WEBCAM_STATIC_URL
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
        predictions = Prediction.objects.filter(params__features__webcam=self)
        return predictions.latest() if predictions else None

    def prediction_params(self):
        return PredictionParams.objects.filter(features__webcam=self)

    def store(self):
        webcam_fs.add_webcam(self.webcam_id)
        with webcam_fs.get_dataset(self.webcam_id) as dataset:
            for featureset in dataset.imgset.feature_sets.keys():
                db_set = FeatureSet(webcam=self, name=featureset)
                db_set.save()

    def post_delete(self):
        webcam_fs.delete_webcam(self.webcam_id)

    def __unicode__(self):
        return self.name


class Picture:
    "Model of a picture taken by a webcam"
    def __init__(self, webcam, timestamp, filep):
        self.webcam = webcam
        self.timestamp = int(timestamp)
        self.filep = filep
        self._pixels = None

    @property
    def pixels(self):
        if self._pixels is None:
            self._pixels = webcam_fs.get_pixels(self.filep,
                                                self.webcam.webcam_id)
        return self._pixels

    def save(self):
        img = webcam_fs.add_picture(self.webcam, self.timestamp, self.filep)
        self._pixels = img['pixels']


class FeatureSet(models.Model):
    "Description of a FeatureSet available in the DataSet"
    RAW = 'raw'
    PCA = 'pca'
    RBM = 'rbm'
    EXTRACT_TYPES = ((RAW, 'Raw'), (PCA, 'PCA'), (RBM, 'RBM'))

    webcam = models.ForeignKey(Webcam)
    extract_type = models.CharField(max_length=16, choices=EXTRACT_TYPES)
    name = models.SlugField(max_length=16)

    def store(self):
        self.name = webcam_fs.add_feature_set(self)

    def post_delete(self):
        webcam_fs.delete_feature_set(self)

    def __unicode__(self):
        return self.webcam.name + ' - ' + self.name


class PredictionParams(models.Model):
    "The parameters for the computation of predictions"
    name = models.SlugField(max_length=16)
    features = models.ForeignKey(FeatureSet)
    intervals = CommaSeparatedIntegerField(max_length=128)

    @property
    def webcam(self):
        return self.features.webcam

    def latest_prediction(self):
        predictions = self.prediction_set
        return predictions.latest() if predictions else None

    def history(self, length=5, orderby='-comp_date', with_error=True):
        "Return the latest predictions for this parameters object"
        history_set = self.prediction_set.order_by(orderby)

        if with_error is True:
            history_set = history_set.exclude(error=None)
        elif with_error is False:
            history_set = history_set.filter(error=None)

        return history_set[:length]

    def error_data(self):
        """
        The complete list of prediction errors for this parameters object,
        ordered by ascending computation date

        Return
        ------
        list(tuple) : The computation dates and associated error
        """
        predictions = self.prediction_set.order_by('comp_date')
        predictions = predictions.exclude(error=None)
        errors = predictions.values_list('comp_date', 'error')
        return errors

    def save(self, *args, **kwargs):
        """
        Create a set of examples in DB and in the dataset of the cam.
        The update of the dataset is done asynchronously as it may take time
        to read all the existing images of the cam.

        FIXME: race conditions with image upload
        FIXME: what if the dataset update fail? the DB is already updated

        Note:
        -----
        This will erase any set with the same name!
        """
        super(PredictionParams, self).save(*args, **kwargs)
        webcam_fs.add_examples_set(self)

    def post_delete(self):
        webcam_fs.delete_examples_set(self)

    def __unicode__(self):
        return '%s.%s: %s' % (
            self.webcam.webcam_id, self.name, str(self.intervals))


class Prediction(models.Model):
    params = models.ForeignKey(PredictionParams)
    comp_date = models.DateTimeField('computation date')
    path = models.CharField(max_length=100)  # FIXME use FileField ?
    error = models.FloatField(null=True)

    class Meta:
        get_latest_by = 'comp_date'

    def url(self):
        return '/%s/%s' % (WEBCAM_STATIC_URL, self.path)

    def target_in_hours(self):
        return (self.params.intervals[-1] // 3600)

    def actual(self):
        """
        Url to the actual picture received
        at the target time of this prediction
        """
        target_delta = timedelta(seconds=self.params.intervals[-1])
        target_date = self.comp_date + target_delta
        target_timestamp = int(time.mktime(target_date.timetuple()))
        cam_id = self.params.webcam.webcam_id
        rel_path = webcam_fs.picture_path(cam_id, target_timestamp)
        return '/%s/%s' % (WEBCAM_STATIC_URL, rel_path)

    def create(self, *args, **kwargs):
        webcam_fs.add_prediction(self)
        self.save(*args, **kwargs)

    def as_picture(self):
        "Picture representation of the prediction"
        cam = self.params.webcam
        timestamp = time.mktime(self.comp_date.timetuple())
        pic = Picture(cam, timestamp, self.path)
        return pic


@receiver(post_delete, sender=PredictionParams)
@receiver(post_delete, sender=FeatureSet)
@receiver(post_delete, sender=Webcam)
def models_post_delete(sender, instance, **kwargs):
    instance.post_delete()
