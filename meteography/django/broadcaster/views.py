import io
import os.path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # FIXME put somewhere more appropriate
import matplotlib.pyplot as plt

from django.http import HttpResponse, HttpResponseNotFound
from django.shortcuts import get_object_or_404, render
from django.utils.timezone import utc
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views.static import serve as serve_file

from meteography.django.broadcaster import forecast
from meteography.django.broadcaster.models import (
    Webcam, Picture, Prediction, PredictionParams)
from meteography.django.broadcaster.settings import WEBCAM_ROOT


def index(request):
    webcams = Webcam.objects.order_by('name')
    context = {
        'webcams': webcams,
    }
    return render(request, 'broadcaster/index.html', context)


def webcam(request, webcam_id):
    webcam = get_object_or_404(Webcam, pk=webcam_id)
    context = {
        'webcam': webcam,
    }
    return render(request, 'broadcaster/webcam.html', context)


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

    # Make a new prediction and save it for each set of prediction params
    params_list = webcam.predictionparams_set.all()
    for params in params_list:
        prediction = forecast.make_prediction(webcam, params, timestamp)

        # Check if there was any prediction targetting this timestamp,
        # and if yes compute the error
        pred_target = params.intervals[-1]
        comp_timestamp = int(timestamp) - pred_target
        comp_date = datetime.fromtimestamp(float(comp_timestamp), utc)
        old_predictions = Prediction.objects.filter(comp_date=comp_date)
        for prediction in old_predictions:
            forecast.update_prediction(prediction, pic)

    return HttpResponse(status=204)


def error_graph(request, webcam_id, pname):
    pred_params = get_object_or_404(PredictionParams,
                                    webcam_id=webcam_id, name=pname)

    errors = pred_params.error_list()
    plt.plot(errors)
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    plt.close()
    response = HttpResponse(content_type='image/png')
    response.write(img_bytes.getvalue())
    return response


def static_pic(request, webcam_id, path):
    # FIXME make production-ready
    return serve_file(request, os.path.join(webcam_id, path), WEBCAM_ROOT)
