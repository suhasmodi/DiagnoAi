from django.apps import AppConfig


class HealthcareWebsiteConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'healthcare_website'

class ChatConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    
    def ready(self):
        # Import the get_assistant function here to avoid circular imports
        from .views import get_assistant
        
        # Initialize the medical assistant when the app starts
        # This is optional but will make the first query faster
        # Note: This might not work correctly in development with auto-reload
        try:
            get_assistant()
        except Exception as e:
            # We don't want to prevent the app from starting if this fails
            import logging
            logging.getLogger(__name__).error(f"Failed to initialize assistant: {e}")