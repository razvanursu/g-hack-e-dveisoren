from django.urls import path

from .views import SignUpView
from .views import ProgramView

urlpatterns = [
    path("signup/", SignUpView.as_view(), name="signup"),
    path("program_page/", ProgramView.as_view(), name="program_page"),
]