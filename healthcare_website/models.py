from django.db import models

# Create your models here.
from django.utils import timezone

class Conversation(models.Model):
    session_id = models.CharField(max_length=100, unique=True,db_index=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Conversation {self.session_id}"

class Message(models.Model):
    MESSAGE_TYPES = (
        ('human', 'Human'),
        ('ai', 'AI'),
    )
    
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    content = models.TextField()
    message_type = models.CharField(max_length=10, choices=MESSAGE_TYPES)
    timestamp = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['timestamp']
    
    def __str__(self):
        return f"{self.message_type.capitalize()} message at {self.timestamp.strftime('%Y-%m-%d %H:%M')}"


class UserProfile_info(models.Model):
    firstname = models.CharField(max_length=50)
    lastname = models.CharField(max_length=50)
    username = models.CharField(max_length=50, unique=True,db_index=True)
    phone = models.CharField(max_length=15, unique=True)
    email = models.EmailField(unique=True)
    birthdate = models.DateField()  # Set a default date
    gender = models.CharField(max_length=10, choices=[("Male", "Male"), ("Female", "Female"), ("Other", "Other")])
    weight = models.FloatField()
    height = models.FloatField()
    address = models.TextField(null=True, blank=True)  # Default value set
    password = models.CharField(max_length=255,db_index=True)  # Store hashed password

    def str(self):
        return self.username  
    


class Doctor(models.Model):
    name = models.CharField(max_length=100)
    gender = models.CharField(max_length=1, choices=[('M', 'Male'), ('F', 'Female')])
    specialization = models.CharField(max_length=50)
    degrees = models.CharField(max_length=100)
    experience_years = models.PositiveIntegerField()
    registration_date = models.DateTimeField(null=True, blank=True)
    phone_number = models.CharField(max_length=15)
    email = models.EmailField()
    address = models.TextField()
    city = models.CharField(max_length=50)
    state = models.CharField(max_length=50)
    pincode = models.CharField(max_length=10)
    consultation_fee = models.PositiveIntegerField()
    is_available = models.BooleanField(default=True)
    bio = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.name} - {self.specialization}" 
    
# Add this to your models.py file

from django.db import models
from django.conf import settings

class Appointment(models.Model):
    APPOINTMENT_STATUS = (
        ('pending', 'Pending'),
        ('confirmed', 'Confirmed'),
        ('cancelled', 'Cancelled'),
        ('completed', 'Completed'),
    )
    
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    doctor = models.ForeignKey('Doctor', on_delete=models.CASCADE)
    patient_name = models.CharField(max_length=100)
    patient_email = models.EmailField()
    patient_phone = models.CharField(max_length=15)
    appointment_date = models.DateField()
    appointment_time = models.TimeField()
    reason = models.TextField(blank=True, null=True)
    status = models.CharField(max_length=20, choices=APPOINTMENT_STATUS, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.patient_name} - {self.doctor.name} - {self.appointment_date}"
    
    class Meta:
        ordering = ['-appointment_date', 'appointment_time']
        
        
