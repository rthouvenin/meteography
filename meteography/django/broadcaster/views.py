import io

from django.contrib.staticfiles.templatetags.staticfiles import static
from django.http import HttpResponse, HttpResponseNotFound
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from meteography.django.broadcaster.models import Webcam, Picture


def index(request):
    webcams = Webcam.objects.order_by('name')

    for webcam in webcams:
        webcam.prediction = {
            'image': static('meteographer/img/noprediction.png'),
        }
    context = {'webcams': webcams}
    return render(request, 'meteographer/index.html', context)


@csrf_exempt
@require_http_methods(['PUT'])
def picture(request, webcam_id, timestamp):
    # check the webcam exists, return 404 if not
    try:
        webcam = Webcam.objects.get(webcam_id=webcam_id)
    except Webcam.DoesNotExist:
        return HttpResponseNotFound("The webcam %s does not exist" % webcam_id)

    # Save the new picture
    img_bytes = io.BytesIO(request.read())
    pic = Picture(webcam, timestamp, img_bytes)
    pic.save()
    return HttpResponse(status=204)
