from django.shortcuts import render

from meteographer.models import Webcam

def index(request):
    webcams = Webcam.objects.order_by('name')
    context = {'webcams': webcams}
    return render(request, 'meteographer/index.html', context)

