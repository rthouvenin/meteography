from django.conf.urls import patterns, url

from meteography.django.broadcaster import views

webcam_url = 'webcams/(?P<webcam_id>[^/]+)'
picture_url = 'pictures/(?P<timestamp>[^/]+)\.(?P<ext>[A-Za-z]+)'

urlpatterns = patterns('',
    url(r'^$', views.index, name='index'),
    url(r'^' + webcam_url + '/' + picture_url + '$',
        views.picture),
)
