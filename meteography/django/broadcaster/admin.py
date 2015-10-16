from django.contrib import admin

from meteography.django.broadcaster.models import Prediction
from meteography.django.broadcaster.models import Webcam


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
    fields = ('webcam_id', 'name')

    def get_readonly_fields(self, request, obj=None):
        if obj is not None:
            return ('webcam_id', )
        else:
            return ()

    def save_model(self, request, webcam, form, change):
        """Init webcam storage before saving the model, in case of create"""
        if not change:
            webcam.store()
        webcam.save()


@admin.register(Prediction)
class PredictionAdmin(ReadOnlyAdmin):
    pass
