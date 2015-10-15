import os.path

from django.core.files.storage import FileSystemStorage
from django.contrib.staticfiles.templatetags.staticfiles import static
from django.http import HttpResponse, HttpResponseNotFound
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from meteography.django.broadcaster.models import Webcam
from meteography.django.broadcaster.settings import WEBCAM_DIR

webcam_fs = FileSystemStorage(location=WEBCAM_DIR)


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
def picture(request, webcam_id, timestamp, ext):
    # check the webcam exists, return 404 if not
    try:
        Webcam.objects.get(webcam_id=webcam_id)
    except Webcam.DoesNotExist:
        return HttpResponseNotFound("The webcam does not exist")

    # store the request content in a file and return 204 (No Content)
    filename = '%s.%s' % (timestamp, ext)
    filepath = os.path.join(webcam_id, 'pics', filename)
    webcam_fs.save(filepath, request)
    return HttpResponse(status=204)
