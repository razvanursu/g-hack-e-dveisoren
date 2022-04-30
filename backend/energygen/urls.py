from django.urls import path
from . import views

urlpatterns = [
    path("", views.createPlot, name="energygen"),
    path("input_coord/", views.get_name, name="input_coord")
]
