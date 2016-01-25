from django.conf.urls import patterns, url

from meteography.django.broadcaster import views
from meteography.django.broadcaster.settings import WEBCAM_URL

app_name = 'broadcaster'

webcam_url = r'^%s/(?P<webcam_id>[^/]+)' % WEBCAM_URL
picture_url = r'%s/pictures/(?P<timestamp>[^/]+)$' % webcam_url
errgraph_url = r'%s/analysis/(?P<pname>[^/]+)/error_graph.png$' % webcam_url
static_url = r'%s/(?P<path>(predictions|pics)/.*)$' % webcam_url

urlpatterns = patterns('',
    url(r'^$', views.index, name='index'),
    url(picture_url, views.picture),
    url(errgraph_url, views.error_graph, name='error_graph'),
    url(static_url, views.static_pic),
    url(webcam_url, views.webcam, name='webcam'),
)
