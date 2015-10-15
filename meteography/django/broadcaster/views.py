import io
import os.path

from django.core.files.storage import FileSystemStorage
from django.contrib.staticfiles.templatetags.staticfiles import static
from django.http import HttpResponse, HttpResponseNotFound
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from PIL import Image

from meteography.django.broadcaster.models import Webcam
from meteography.django.broadcaster.settings import WEBCAM_DIR, WEBCAM_SIZE

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
def picture(request, webcam_id, timestamp):
    # check the webcam exists, return 404 if not
    try:
        Webcam.objects.get(webcam_id=webcam_id)
    except Webcam.DoesNotExist:
        return HttpResponseNotFound("The webcam does not exist")

    # read and resize the image
    img_bytes = io.BytesIO(request.read())
    img = Image.open(img_bytes)
    if img.size != WEBCAM_SIZE:
        img = img.resize(WEBCAM_SIZE)

    # store the the image and return 204 (No Content)
    filename = '%s.jpg' % timestamp
    filepath = os.path.join(webcam_id, 'pics', filename)
    with webcam_fs.open(filepath, mode='wb') as fp:
        img.save(fp)

    return HttpResponse(status=204)
