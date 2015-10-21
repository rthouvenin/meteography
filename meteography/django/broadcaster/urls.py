from django.conf.urls import patterns, url

from meteography.django.broadcaster import views
from meteography.django.broadcaster.settings import WEBCAM_URL

webcam_url = r'^%s/(?P<webcam_id>[^/]+)' % WEBCAM_URL
picture_url = r'%s/pictures/(?P<timestamp>[^/]+)$' % webcam_url
prediction_url = r'%s/(?P<path>predictions/.*)$' % webcam_url

urlpatterns = patterns('',
    url(r'^$', views.index, name='index'),
    url(picture_url, views.picture),
    url(prediction_url, views.prediction),
)
