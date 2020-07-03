from django.conf.urls import url
from django.urls import path, include
from . import views
from django.contrib.auth.views import LoginView, LogoutView


urlpatterns=[
    
    url(r'',views.home,name='home'),
]