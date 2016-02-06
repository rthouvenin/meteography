import io
from datetime import datetime

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.dates import DateFormatter
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np

from django.http import (
    HttpResponse, HttpResponseNotFound, HttpResponseForbidden)
from django.shortcuts import get_object_or_404, render
from django.utils.timezone import utc
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from meteography.django.broadcaster import forecast
from meteography.django.broadcaster.models import (
    Webcam, Picture, Prediction, PredictionParams)


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
        'histories': [{
            'id': 'hist-latest',
            'name': "Latest",
            'orderby': '-comp_date',
        }, {
            'id': 'hist-worst',
            'name': "Highest error",
            'orderby': '-error',
        }, {
            'id': 'hist-best',
            'name': "Lowest error",
            'orderby': 'error',
        }, {
            'id': 'hist-random',
            'name': "Random",
            'orderby': '?',
        }],
    }
    return render(request, 'broadcaster/webcam.html', context)


@csrf_exempt
@require_http_methods(['PUT'])
def picture(request, webcam_id, timestamp):
    "Handles upload of pictures"
    # FIXME have proper authentication
    hostname = request.get_host()
    has_port = hostname.find(':')
    if has_port > 0:
        hostname = hostname[:has_port]
    if hostname != '127.0.0.1':
        return HttpResponseForbidden()

    # check the webcam exists, return 404 if not
    try:
        webcam = Webcam.objects.get(webcam_id=webcam_id)
    except Webcam.DoesNotExist:
        return HttpResponseNotFound("The webcam %s does not exist" % webcam_id)

    timestamp = int(timestamp)
    # Save the new picture
    img_bytes = io.BytesIO(request.read())
    pic = Picture(webcam, timestamp, img_bytes)
    pic.save()

    # Make a new prediction and save it for each set of prediction params
    params_list = webcam.prediction_params()
    for params in params_list:
        prediction = forecast.make_prediction(webcam, params, timestamp)

        # Check if there was any prediction targetting this timestamp,
        # and if yes compute the error
        pred_target = params.intervals[-1]
        comp_timestamp = timestamp - pred_target
        comp_date = datetime.fromtimestamp(float(comp_timestamp), utc)
        old_predictions = Prediction.objects.filter(comp_date=comp_date,
                                                    params=params)
        for prediction in old_predictions:
            forecast.update_prediction(prediction, pic)

    return HttpResponse(status=204)


def error_graph(request, webcam_id, pname):
    "Generate a graph of error evolution over time"
    pred_params = get_object_or_404(PredictionParams,
                                    features__webcam_id=webcam_id, name=pname)

    error_data = pred_params.error_data()
    if error_data:
        dates, errors = zip(*error_data)
        # Smoothen data with moving average
        window = 12
        errors = np.convolve(errors, np.ones(window)/window, mode='same')

    # Generate the graph with matplotlib
    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_title("Evolution of error value over time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Error")
    if error_data:
        ax.plot(dates, errors)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_formatter(DateFormatter('%d/%m/%Y'))

    # Write the graph image to the response
    response = HttpResponse(content_type='image/png')
    fig.savefig(response, format='png')

    return response
