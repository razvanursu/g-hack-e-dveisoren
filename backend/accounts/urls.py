from django.urls import path

from .views import SignUpView

urlpatterns = [
    path("signup/", SignUpView.as_view(), name="signup"),
    #path('loggedin', SignUpView.as_view(template_name='start.html'), name='loggedin'),
]