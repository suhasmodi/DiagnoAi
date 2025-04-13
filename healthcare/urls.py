"""
URL configuration for healthcare project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from healthcare_website.views import *

from django.conf import settings
from django.conf.urls.static import static
from healthcare_website import views  # Add this import




urlpatterns = [
    path('admin/', admin.site.urls),
    path("index/", index, name="index"),
    path("diseases-predict/",disease_prediction,name='disease_prediction'),
    path('chat/', show_chat, name='show_chat'),
    path('api/query/', process_query, name='process_query'),
    path('api/reset/', reset_conversation, name='reset_conversation'),
    path('analyze/', AnalyzeReport, name='AnalyzeReport'),
    path("signup/", signup, name="signup"),
    path("", signin, name="signin"),
    path("news/",news,name='news'),
    path('get-doctors/', views.get_doctors, name='get_doctors'),
    path('book-appointment/', views.book_appointment, name='book_appointment'),
    
]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

