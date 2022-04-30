import imp
from django.shortcuts import render, redirect
from django.urls import reverse
import plotly.io as io
import plotly.express as px
import os
from django.http import HttpResponse, HttpResponseRedirect

from .forms import NameForm

from .predictor.linear_predictor import predict_renew_location

import numpy as np
import pandas as pd


# Create your views here.


def createPlot(request, prediction):
    fig = px.plot(x=range(24), y=prediction)
    plot = io.to_html(fig)
    # print(plot)
    return HttpResponse(plot)

# def mathias_func(x):
#     return y


def get_name(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = NameForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            latitude = form.cleaned_data['latitude']
            longitude = form.cleaned_data['longitude']
            time = form.cleaned_data['time']
            country = form.cleaned_data['country']

            print(latitude, longitude, time)

            prediction = list(predict_renew_location(latitude, longitude, time, country))

            array = np.array([list(range(24)), prediction])
            df = pd.DataFrame(array.transpose(), columns=["Time [h]", "Percentage [%]"])
            fig = px.line(df, x = "Time [h]", y = "Percentage [%]",
                title=f"Predicted renewable energy share for your location for the next 24 hours")
            fig.add_hline(y=np.mean(prediction))
            plot = io.to_html(fig)
            # print(plot)
            return HttpResponse(plot)

    # if a GET (or any other method) we'll create a blank form
    else:
        form = NameForm()
    
    # print(form)

    return render(request, 'energygen/input_coord.html', {'form': form})
