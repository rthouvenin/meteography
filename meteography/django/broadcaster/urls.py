from django.conf import settings as django_settings
from django.conf.urls import patterns, url, static

from meteography.django.broadcaster import views
from meteography.django.broadcaster.settings import (
    WEBCAM_URL, WEBCAM_ROOT, WEBCAM_STATIC_URL)


app_name = 'broadcaster'

webcam_url = r'^%s/(?P<webcam_id>[^/]+)' % WEBCAM_URL
picture_url = r'%s/pictures/(?P<timestamp>[^/]+)$' % webcam_url
errgraph_url = r'%s/analysis/(?P<pname>[^/]+)/error_graph.png$' % webcam_url

urlpatterns = patterns('',
    url(r'^$', views.index, name='index'),
    url(picture_url, views.picture),
    url(errgraph_url, views.error_graph, name='error_graph'),
    url(webcam_url, views.webcam, name='webcam'),
)

if django_settings.DEBUG is True:
    urlpatterns += static.static(WEBCAM_STATIC_URL, document_root=WEBCAM_ROOT)
