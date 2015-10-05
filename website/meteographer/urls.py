from django.conf.urls import patterns, url

from meteographer import views

urlpatterns = patterns('',
    url(r'^$', views.index, name='index')
)

