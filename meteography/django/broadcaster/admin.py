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
        

class WebcamAdmin(admin.ModelAdmin):
    list_display = ('name', )

class PredictionAdmin(ReadOnlyAdmin):
    pass
    
admin.site.register(Webcam, WebcamAdmin)
admin.site.register(Prediction, PredictionAdmin)
