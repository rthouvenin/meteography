from django.shortcuts import render
from django.contrib.staticfiles.templatetags.staticfiles import static

from meteography.django.broadcaster.models import Webcam

def index(request):
    webcams = Webcam.objects.order_by('name')
    
    for webcam in webcams:
        webcam.prediction = {
            'image': static('meteographer/img/noprediction.png'),
        }
    context = {'webcams': webcams}
    return render(request, 'meteographer/index.html', context)

