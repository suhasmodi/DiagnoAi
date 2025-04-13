from django.contrib import admin
from .models import Conversation, Message
from .models import UserProfile_info,Doctor
# Import your models

@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ['session_id', 'created_at', 'updated_at']  # Fields to display in admin panel
    search_fields = ['session_id']  # Add search functionality

@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ['conversation', 'message_type', 'timestamp']  # Fields to display
    list_filter = ['message_type', 'timestamp']  # Add filters
    search_fields = ['content']  # Allow search on message content


class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['username', 'firstname', 'lastname', 'email', 'phone', 'birthdate', 'gender', 'weight', 'height']
    search_fields = ['username', 'email', 'phone']
    
admin.site.register(UserProfile_info, UserProfileAdmin)



@admin.register(Doctor)
class DoctorAdmin(admin.ModelAdmin):
    list_display = ('name', 'specialization', 'degrees', 'experience_years', 'city', 'consultation_fee', 'is_available')
    list_filter = ('specialization', 'is_available', 'city', 'state')
    search_fields = ('name', 'email', 'phone_number', 'specialization')
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'gender', 'specialization', 'degrees', 'experience_years', 'registration_date')
        }),
        ('Contact Details', {
            'fields': ('phone_number', 'email', 'address', 'city', 'state', 'pincode')
        }),
        ('Professional Details', {
            'fields': ('consultation_fee', 'is_available', 'bio')
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    list_editable = ('is_available', 'consultation_fee')
    list_per_page = 20
    
    def get_ordering(self, request):
        return ['-created_at']  # Order by most recently created first