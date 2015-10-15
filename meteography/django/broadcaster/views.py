import io

from django.contrib.staticfiles.templatetags.staticfiles import static
from django.http import HttpResponse, HttpResponseNotFound
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from meteography.django.broadcaster.models import Webcam
from meteography.django.broadcaster.storage import WebcamStorage

webcam_fs = WebcamStorage()


def index(request):
    webcams = Webcam.objects.order_by('name')

    for webcam in webcams:
        webcam.prediction = {
            'image': static('meteographer/img/noprediction.png'),
        }
    context = {'webcams': webcams}
    return render(request, 'meteographer/index.html', context)


def webcam(request, webcam_id):
    pass


@csrf_exempt
@require_http_methods(['PUT'])
def picture(request, webcam_id, timestamp):
    # check the webcam exists, return 404 if not
    try:
        Webcam.objects.get(webcam_id=webcam_id)
    except Webcam.DoesNotExist:
        return HttpResponseNotFound("The webcam does not exist")

    img_bytes = io.BytesIO(request.read())
    webcam_fs.add_picture(webcam_id, timestamp, img_bytes)

    return HttpResponse(status=204)
