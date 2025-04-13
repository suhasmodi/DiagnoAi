# add_doc_info.py
import os
import sys
import django
import random
from faker import Faker

# Setup Django environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'healthcare.settings')  # Change 'healthcare' if your project has a different name
django.setup()

# Now you can import your models
from healthcare_website.models import Doctor  # Update this import to match your model location

def generate_doctors(count):
    fake = Faker('en_IN')  # Use Indian locale
    
    # Common Indian surnames
    indian_surnames = [
        'Sharma', 'Patel', 'Verma', 'Gupta', 'Singh', 'Kumar', 'Agarwal', 'Mehta',
        'Joshi', 'Shah', 'Rao', 'Reddy', 'Patil', 'Desai', 'Nair', 'Iyer', 'Iyengar',
        'Chatterjee', 'Banerjee', 'Mukherjee', 'Bose', 'Das', 'Malhotra', 'Kapoor',
        'Chauhan', 'Chopra', 'Dutta', 'Gill', 'Bhat', 'Pillai', 'Kaur', 'Arora'
    ]
    
    # Medical specializations
    specializations = [
        'Cardiology', 'Neurology', 'Pediatrics', 'Orthopedics', 'Gynecology',
        'Dermatology', 'Ophthalmology', 'Psychiatry', 'Oncology', 'Urology',
        'Gastroenterology', 'Pulmonology', 'Nephrology', 'Endocrinology',
        'Rheumatology', 'Hematology', 'Infectious Disease', 'General Surgery',
        'Anesthesiology', 'Radiology', 'Pathology', 'Family Medicine',
        'Internal Medicine', 'Ayurveda', 'Homeopathy', 'Unani Medicine'
    ]
    
    # Degrees
    degrees = ['MBBS', 'MD', 'MS', 'DNB', 'DM', 'MCh', 'BAMS', 'BHMS', 'BUMS']

    doctors_created = 0
    doctors_to_create = []
    fees=[300,400,500,450,550,700,800,900,1050,1500]
    for _ in range(count):
        # Generate a more Indian-sounding name
        gender = random.choice(['M', 'F'])
        if gender == 'M':
            first_name = fake.first_name_male()
        else:
            first_name = fake.first_name_female()
            
        last_name = random.choice(indian_surnames)
        
        # Generate 1-3 random degrees
        doctor_degrees = ', '.join(random.sample(degrees, random.randint(1, 3)))
        
        # Create the doctor object
        doctor = Doctor(
            name=f"Dr. {first_name} {last_name}",
            gender=gender,
            specialization=random.choice(specializations),
            degrees=doctor_degrees,
            experience_years=random.randint(1, 35),
            phone_number=fake.phone_number(),
            email=f"{first_name.lower()}.{last_name.lower()}@{fake.domain_name()}",
            address=fake.address(),
            city=fake.city(),
            state=fake.state(),
            pincode=fake.postcode(),
            consultation_fee=random.choice(fees),
            is_available=random.choice([True, False]),
            bio=fake.paragraph(nb_sentences=5),
        )
        
        doctors_to_create.append(doctor)
        doctors_created += 1
        
        if doctors_created % 50 == 0:
            # Save in batches of 50 for better performance
            Doctor.objects.bulk_create(doctors_to_create)
            print(f'Created and saved {doctors_created} doctors so far...')
            doctors_to_create = []
    
    # Save any remaining doctors
    if doctors_to_create:
        Doctor.objects.bulk_create(doctors_to_create)
        
    print(f'Successfully created {doctors_created} fake Indian doctors and stored them in the database!')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        count = int(sys.argv[1])
        generate_doctors(count)
    else:
        generate_doctors(50)  # Default count