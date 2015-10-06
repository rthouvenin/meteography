from django.conf.urls import patterns, url

from meteography.django.broadcaster import views

urlpatterns = patterns('',
    url(r'^$', views.index, name='index')
)

