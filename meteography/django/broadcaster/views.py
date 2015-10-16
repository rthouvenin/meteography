import io

import matplotlib.pylab as plt

from django.contrib.staticfiles.templatetags.staticfiles import static
from django.http import HttpResponse, HttpResponseNotFound
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from meteography.dataset import DataSet
from meteography.neighbors import NearestNeighbors
from meteography.django.broadcaster.models import Webcam, Picture
from meteography.django.broadcaster.storage import WebcamStorage


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

    webcam_fs = WebcamStorage()
    hdf5_path = webcam_fs.fs.path(webcam_fs.dataset_path(webcam_id))
    with DataSet.open(hdf5_path) as dataset:
        onlineset = dataset.get_set('online')
        new_input = dataset.make_input(onlineset, int(timestamp))
        if new_input is not None and len(onlineset.input) > 0:
            neighbors = NearestNeighbors()
            neighbors.fit(onlineset.input, onlineset.output)
            output = neighbors.predict(new_input).reshape((60, 80, 3))
            plt.imsave(webcam_fs.fs.path('prediction.jpg'), output)

    return HttpResponse(status=204)
