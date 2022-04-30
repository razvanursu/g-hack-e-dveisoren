from django.shortcuts import render
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.views import generic
from .energyPredictor import createPredictionPlot

class energyView(generic.CreateView):
    form_class = UserCreationForm
    success_url = reverse_lazy("energyGen")
    createPredictionPlot()
    template_name = "energyGen/energyPlot.html"