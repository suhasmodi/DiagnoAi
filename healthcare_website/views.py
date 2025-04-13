from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from django.shortcuts import render
from .utils import predict_disease, get_disease_information
from .models import Conversation, Message
from .ai_chat import MedicalAssistant
import json
import uuid
import threading
from .forms import UploadPDFForm
import os
import pytesseract
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_together import ChatTogether

import os
from django.conf import settings
from django.http import JsonResponse
from .forms import UploadPDFForm

from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.hashers import check_password
from healthcare_website.models import UserProfile_info

from django.contrib import messages
from django.contrib.auth.hashers import make_password
from healthcare_website.models import UserProfile_info


def index(request):
    
    specializations = Doctor.objects.values_list('specialization', flat=True).distinct()
    
    
    specializations_list = list(specializations)
    
    
    print(f"Available specializations: {specializations_list}")
    
    context = {
        'specializations': specializations_list,
        
    }
    
    return render(request, 'index.html', context)






AVAILABLE_SYMPTOMS = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety','cold_hands_and_feets','mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat',
'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']

def disease_prediction(request):
    
    context = {'available_symptoms': AVAILABLE_SYMPTOMS}
    
    if request.method == 'POST':
        
        symptoms_string = request.POST.get('symptoms', '')
        selected_symptoms = [symptom.strip() for symptom in symptoms_string.split(',') if symptom.strip()]
        
        print("Received symptoms string:", symptoms_string)
        print("Processed symptoms list:", selected_symptoms)
        
        if selected_symptoms:
            
            predicted_disease = predict_disease(selected_symptoms)
            print("Predicted disease:", predicted_disease)
            
            
            (disease_description,
             disease_precautions,
             disease_medications,
             disease_diet,
             disease_workout,
             doc_dis) = get_disease_information(predicted_disease)
            
            
            context.update({
                'selected_symptoms': selected_symptoms,
                'predicted_disease': predicted_disease,
                'disease_description': disease_description,
                
                'disease_precautions': [f"{i + 1}. {precaution}" for i, precaution in enumerate(disease_precautions[0])],
                
                'disease_medications': [f"{i + 1}. {medication}" for i, medication in enumerate(disease_medications)],
                'disease_diet': [f"{i + 1}. {diet}" for i, diet in enumerate(disease_diet)],
                'disease_workout': [f"{i + 1}. {workout}" for i, workout in enumerate(disease_workout)],
                'doc_dis': [f"{i + 1}. {doctor}" for i, doctor in enumerate(doc_dis)],
            })
        else:
            
            context['error'] = 'Please select at least one symptom.'
    
    
    return render(request, 'predict-diseases.html', context)



    


medical_assistant = None
assistant_lock = threading.Lock()

def get_assistant():
    global medical_assistant
    with assistant_lock:
        if medical_assistant is None:
            medical_assistant = MedicalAssistant()
    return medical_assistant

def show_chat(request):
    """Render the main chat interface"""
    
    if 'session_id' not in request.session:
        request.session['session_id'] = str(uuid.uuid4())
    
    
    session_id = request.session['session_id']
    conversation, created = Conversation.objects.get_or_create(session_id=session_id)
    
    
    messages = Message.objects.filter(conversation=conversation)
    
    return render(request, 'chat.html', {
        'messages': messages,
        'session_id': session_id
    })

@csrf_exempt
def process_query(request):
    """Process a chat query and return the response"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_input = data.get('query', '')
            
            
            session_id = request.session.get('session_id', str(uuid.uuid4()))
            request.session['session_id'] = session_id
            
            
            conversation, created = Conversation.objects.get_or_create(session_id=session_id)
            
            
            Message.objects.create(
                conversation=conversation,
                content=user_input,
                message_type='human'
            )
            
            
            assistant = get_assistant()
            response = assistant.process_query(user_input)
            
            
            Message.objects.create(
                conversation=conversation,
                content=response['answer'],
                message_type='ai'
            )
            
            return JsonResponse({
                'status': 'success',
                'answer': response['answer'],
                'sources': response.get('sources', []),
                'fallback': response.get('fallback', False)
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Only POST requests are allowed'}, status=405)

@csrf_exempt
def reset_conversation(request):
    """Reset the conversation history"""
    if request.method == 'POST':
        
        assistant = get_assistant()
        assistant.reset_conversation()
        
        
        session_id = request.session.get('session_id')
        if session_id:
            try:
                conversation = Conversation.objects.get(session_id=session_id)
                conversation.messages.all().delete()
            except Conversation.DoesNotExist:
                pass
        
        return JsonResponse({'status': 'success', 'message': 'Conversation reset successfully'})
    
    return JsonResponse({'status': 'error', 'message': 'Only POST requests are allowed'}, status=405)



 

def AnalyzeReport(request):
    if request.method == 'POST':
        print("Request Received")
        print("FILES:", request.FILES)

        
        if 'file' not in request.FILES:
            return JsonResponse({"error": "No file uploaded"})
        
        file = request.FILES['file']
        print("File Name:", file.name)

        
        if not file.name.lower().endswith('.pdf'):
            return JsonResponse({"error": "Only PDF files are supported"})

        
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'temp_uploads')
        os.makedirs(upload_dir, exist_ok=True)
        unique_filename = f"{uuid.uuid4()}_{file.name}"
        file_path = os.path.join(upload_dir, unique_filename)

        try:
            
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            print(f"File saved successfully at {file_path}")
            
            
            loader = PyPDFLoader(file_path)
            doc = loader.load()
            if not doc or len(doc) == 0 or not doc[0].page_content:
                return JsonResponse({"error": "No text content found in the PDF"})
            
            text = doc[0].page_content
            print("Extracted Text (first 100 chars):", text[:100])
            
            
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base", chunk_size=300, chunk_overlap=50
            )
            texts = text_splitter.split_text(text)
            
            
            prompt_template = """
            You are a medical report analysis assistant. Provide a summary of the medical report:
            "{text}"
            Summary:
            - Key Findings:
            - Health Condition Overview:
            - Recommendations:
            """
            prompt = PromptTemplate.from_template(prompt_template)
            
            
            llm_model = ChatTogether(
                together_api_key=settings.TOGETHER_API_KEY,
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            )
            llm_chain = LLMChain(llm=llm_model, prompt=prompt)
            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
            
            
            summary = stuff_chain.run(doc)
            print("Summary:", summary)
            return JsonResponse({"summary": summary})

        except Exception as e:
            print("Error processing PDF:", str(e))
            return JsonResponse({"error": f"Error processing PDF: {str(e)}"})

        finally:
            
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {unique_filename} removed successfully")
    
    return JsonResponse({"error": "Invalid request method. Use POST."})


  

def signup(request):
    if request.method == "POST":
        firstname = request.POST["firstname"]
        lastname = request.POST["lastname"]
        username = request.POST["username"]
        phone = request.POST["phone"]
        email = request.POST["email"]
        birthdate = request.POST["birthdate"]  
        gender = request.POST["gender"]
        weight = request.POST["weight"]
        height = request.POST["height"]
        address = request.POST["address"]
        password = request.POST["password"]

        
        if UserProfile_info.objects.filter(username=username).exists():
            messages.error(request, "Username already exists! Please choose another.")
            return redirect("signup")

        if UserProfile_info.objects.filter(email=email).exists():
            messages.error(request, "Email already registered! Please use another email.")
            return redirect("signup")

        
        hashed_password = make_password(password)

        
        user = UserProfile_info.objects.create(
            firstname=firstname,
            lastname=lastname,
            username=username,
            phone=phone,
            email=email,
            birthdate=birthdate,  
            gender=gender,
            weight=weight,
            height=height,
            address=address,
            password=hashed_password,
        )

        messages.success(request, "Signup successful! You can now login.")
        return redirect("signin")

    return render(request, "signup.html")

  

from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.shortcuts import render, redirect
from django.core.cache import cache


def signin(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        
        
        user = UserProfile_info.objects.filter(username=username).first()
        if not user:
            messages.error(request, "Username does not exist. Please sign up.")
            return redirect("signin")
            
        if check_password(password, user.password):
            
            request.session['user_id'] = user.id
            request.session['username'] = user.username
            return redirect("index")
        else:
            messages.error(request, "Incorrect password. Try again.")
            return redirect("signin")
            
    return render(request, "signin.html")



def news(request):
    return render(request,'final3.html')


from django.core.management.base import BaseCommand
from faker import Faker
import random
from .models import Doctor  

class Command(BaseCommand):
    help = 'Creates fake Indian doctor data'

    def add_arguments(self, parser):
        parser.add_argument('count', type=int, help='Number of fake doctors to create')

    def handle(self, *args, **kwargs):
        count = kwargs['count']
        fake = Faker('en_IN')  
        
        
        indian_surnames = [
            'Sharma', 'Patel', 'Verma', 'Gupta', 'Singh', 'Kumar', 'Agarwal', 'Mehta',
            'Joshi', 'Shah', 'Rao', 'Reddy', 'Patil', 'Desai', 'Nair', 'Iyer', 'Iyengar',
            'Chatterjee', 'Banerjee', 'Mukherjee', 'Bose', 'Das', 'Malhotra', 'Kapoor',
            'Chauhan', 'Chopra', 'Dutta', 'Gill', 'Bhat', 'Pillai', 'Kaur', 'Arora'
        ]
        
        
        specializations = [
            'Cardiology', 'Neurology', 'Pediatrics', 'Orthopedics', 'Gynecology',
            'Dermatology', 'Ophthalmology', 'Psychiatry', 'Oncology', 'Urology',
            'Gastroenterology', 'Pulmonology', 'Nephrology', 'Endocrinology',
            'Rheumatology', 'Hematology', 'Infectious Disease', 'General Surgery',
            'Anesthesiology', 'Radiology', 'Pathology', 'Family Medicine',
            'Internal Medicine', 'Ayurveda', 'Homeopathy', 'Unani Medicine'
        ]
        
        
        degrees = ['MBBS', 'MD', 'MS', 'DNB', 'DM', 'MCh', 'BAMS', 'BHMS', 'BUMS']

        doctors_created = 0
        
        for _ in range(count):
            
            gender = random.choice(['M', 'F'])
            if gender == 'M':
                first_name = fake.first_name_male()
            else:
                first_name = fake.first_name_female()
                
            last_name = random.choice(indian_surnames)
            
            
            doctor_degrees = ', '.join(random.sample(degrees, random.randint(1, 3)))
            
            
            doctor = Doctor(
                name=f"Dr. {first_name} {last_name}",
                gender=gender,
                specialization=random.choice(specializations),
                degrees=doctor_degrees,
                experience_years=random.randint(1, 35),
                phone_number=fake.phone_number(),
                email=fake.email(),
                address=fake.address(),
                city=fake.city(),
                state=fake.state(),
                pincode=fake.postcode(),
                consultation_fee=random.randint(300, 3000),
                is_available=random.choice([True, False]),
            )
            
            doctor.save()
            doctors_created += 1
            
            if doctors_created % 10 == 0:
                self.stdout.write(f'Created {doctors_created} doctors so far...')
        
        self.stdout.write(self.style.SUCCESS(f'Successfully created {doctors_created} fake Indian doctors!'))
        


from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.conf import settings
from .models import Doctor, Appointment
import json
from datetime import datetime

def home(request):
    """Home page view that provides specializations for the appointment form"""
    return render(request, 'index.html')

def get_doctors(request):
    specialization = request.GET.get('specialization', '')
    
    print(f"Received request for doctors with specialization: {specialization}")
    
    if specialization:
        doctors = Doctor.objects.filter(specialization=specialization).values('id', 'name')
        doctors_list = list(doctors)
        
        print(f"Found doctors: {doctors_list}")
        
        return JsonResponse({'doctors': doctors_list})
    
    print("No specialization provided or empty specialization")
    return JsonResponse({'doctors': []})

@require_POST

def book_appointment(request):
    if request.method == 'POST':
        
        doctor_id = request.POST.get('doctor')
        patient_name = request.POST.get('patient_name')
        patient_email = request.POST.get('patient_email')
        patient_phone = request.POST.get('patient_phone')
        appointment_date = request.POST.get('appointment_date')
        appointment_time = request.POST.get('appointment_time')
        reason = request.POST.get('reason')
        
        try:
            
            doctor = Doctor.objects.get(id=doctor_id)
            
            appointment = Appointment.objects.create(
                doctor=doctor,
                patient_name=patient_name,
                patient_email=patient_email,
                patient_phone=patient_phone,
                appointment_date=appointment_date,
                appointment_time=appointment_time,
                reason=reason
            )
            
            
            email_sent = send_confirmation_emails(appointment)
            
            return JsonResponse({
                'status': 'success',
                'message': 'Appointment booked successfully!' + (' Email confirmation sent.' if email_sent else ' Email could not be sent.')
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    })

def send_confirmation_emails(appointment):
    """Send confirmation emails to both patient and doctor"""
    
    
    try:
        
        from datetime import datetime
        
        if isinstance(appointment.appointment_date, str):
            date_obj = datetime.strptime(appointment.appointment_date, '%Y-%m-%d')
            formatted_date = date_obj.strftime('%A, %B %d, %Y')
        else:
            formatted_date = appointment.appointment_date.strftime('%A, %B %d, %Y')
            
        if isinstance(appointment.appointment_time, str):
            time_obj = datetime.strptime(appointment.appointment_time, '%H:%M')
            formatted_time = time_obj.strftime('%I:%M %p')
        else:
            formatted_time = appointment.appointment_time.strftime('%I:%M %p')
    except Exception as e:
        
        print(f"Error formatting date/time: {str(e)}")
        formatted_date = appointment.appointment_date
        formatted_time = appointment.appointment_time
    
    
    context = {
        'appointment': appointment,
        'doctor_name': appointment.doctor.name,
        'patient_name': appointment.patient_name,
        'appointment_date': formatted_date,
        'appointment_time': formatted_time,
        'reason': appointment.reason
    }
    
    
    patient_subject = f'Appointment Confirmation with {appointment.doctor.name}'
    patient_html_message = render_to_string('pateint_confirmation.html', context)
    patient_plain_message = strip_tags(patient_html_message)
    
    
    doctor_subject = f'New Appointment with {appointment.patient_name}'
    doctor_html_message = render_to_string('doctor_notification.html', context)
    doctor_plain_message = strip_tags(doctor_html_message)
    
    
    try:
        
        send_mail(
            patient_subject,
            patient_plain_message,
            settings.DEFAULT_FROM_EMAIL,
            [appointment.patient_email],
            html_message=patient_html_message,
            fail_silently=False
        )
        
        
        if hasattr(appointment.doctor, 'email') and appointment.doctor.email:
            send_mail(
                doctor_subject,
                doctor_plain_message,
                settings.DEFAULT_FROM_EMAIL,
                [appointment.doctor.email],
                html_message=doctor_html_message,
                fail_silently=False
            )
        else:
            print("Doctor email not available, skipping doctor notification")
        
        return True
    except Exception as e:
        print(f"Error sending emails: {str(e)}")
        return False