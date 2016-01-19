from django.contrib import admin

from meteography.django.broadcaster.models import (
    Webcam, PredictionParams, Prediction)


admin.site.site_header = "Meteography administration"


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
    fieldsets = (
        (None, {'fields': ('webcam_id', 'name')}),
        ('Compression', {
            'fields': ('compressed', ),
            'description': """Compression can be used to reduce the space taken
                by the webcam data and improve computation speed, but may
                reduce the quality of the predictions.""",
        }),
    )

    def get_readonly_fields(self, request, obj=None):
        if obj is not None:
            return ('webcam_id', )
        else:
            return ()

    def save_model(self, request, webcam, form, change):
        """Init webcam storage before saving the model, in case of create"""
        if not change:
            webcam.store()
        elif webcam.compressed:
            # FIXME improve workflow
            webcam.compress()
        webcam.save()


@admin.register(PredictionParams)
class PredictionParamsAdmin(admin.ModelAdmin):
    list_display = ('webcam', 'name', 'intervals')
    list_display_links = ('name', )
    fields = ('webcam', 'name', 'intervals')

    def get_readonly_fields(self, request, obj=None):
        if obj is not None:
            return ('webcam', 'name')
        else:
            return ()


@admin.register(Prediction)
class PredictionAdmin(ReadOnlyAdmin):
    list_display = ('params', 'comp_date', 'path')
    list_display_links = ('comp_date', )
    pass
