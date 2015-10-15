import os.path

from django.contrib import admin

from meteography.django.broadcaster.models import Prediction
from meteography.django.broadcaster.models import Webcam
from meteography.django.broadcaster.settings import WEBCAM_DIR, WEBCAM_SIZE

from meteography.dataset import ImageSet, DataSet


class ReadOnlyAdmin(admin.ModelAdmin):
    """A base class for read-only models"""
    readonly_fields = []

    def get_readonly_fields(self, request, obj=None):
        return list(self.readonly_fields) + \
            [field.name for field in obj._meta.fields]

    def has_add_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(Webcam)
class WebcamAdmin(admin.ModelAdmin):
    list_display = ('name', )

    def save_model(self, request, webcam, form, change):
        hdf5_path = os.path.join(WEBCAM_DIR, webcam.webcam_id + '.h5')
        img_shape = WEBCAM_SIZE[1], WEBCAM_SIZE[0], 3
        imageset = ImageSet.create(hdf5_path, img_shape)
        dataset = DataSet.create(imageset.fileh, imageset)
        dataset.close()
        os.makedirs(os.path.join(WEBCAM_DIR, webcam.webcam_id, 'pics'))
        webcam.save()


@admin.register(Prediction)
class PredictionAdmin(ReadOnlyAdmin):
    pass
