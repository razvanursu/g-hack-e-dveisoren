from django.urls import path
from .views import energyView

from . import views

urlpatterns = [
    path('energyGen/', energyView.as_view(), name='energyGen'),
]