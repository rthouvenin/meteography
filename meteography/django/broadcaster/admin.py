from django.contrib import admin

from meteography.django.broadcaster.models import Prediction
from meteography.django.broadcaster.models import Webcam
from meteography.django.broadcaster.storage import WebcamStorage


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
        """Init webcam storage before saving the model"""
        webcam_fs = WebcamStorage()
        webcam_fs.add_webcam(webcam.webcam_id)
        webcam.save()


@admin.register(Prediction)
class PredictionAdmin(ReadOnlyAdmin):
    pass
