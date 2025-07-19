import os
import ast
import pandas as pd
import numpy as np
import joblib
import logging
from django.conf import settings
from fuzzywuzzy import process
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


dataset_dir = os.path.join(settings.BASE_DIR, "healthcare_website", "dataset")
description = pd.read_csv(os.path.join(dataset_dir, "description.csv"))
precautions = pd.read_csv(os.path.join(dataset_dir, "precautions_df.csv"))
medications = pd.read_csv(os.path.join(dataset_dir, "medications.csv"))
diets = pd.read_csv(os.path.join(dataset_dir, "diets.csv"))
doc_vs_dis = pd.read_csv(os.path.join(dataset_dir, "Doctor_Versus_Disease.csv"), encoding='latin1')

workout = pd.read_csv(os.path.join(dataset_dir, "workout_df.csv"))


# Symptom list used for fuzzy matching
symptoms = [
    'abdominal_pain', 'abnormal_menstruation', 'acidity', 'acute_liver_failure',
    'altered_sensorium', 'anxiety', 'back_pain', 'belly_pain', 'blackheads',
    'bladder_discomfort', 'blister', 'blood_in_sputum', 'bloody_stool', 'blurred_and_distorted_vision',
    'breathlessness', 'brittle_nails', 'bruising', 'burning_micturition', 'chest_pain',
    'chills', 'cold_hands_and_feets', 'coma', 'congestion', 'constipation', 'continuous_feel_of_urine',
    'continuous_sneezing', 'cough', 'cramps', 'dark_urine', 'dehydration', 'depression',
    'diarrhoea', 'dischromic_patches', 'distention_of_abdomen', 'dizziness', 'drying_and_tingling_lips',
    'enlarged_thyroid', 'excessive_hunger', 'extra_marital_contacts', 'family_history',
    'fast_heart_rate', 'fatigue', 'fluid_overload', 'foul_smell_of_urine', 'headache',
    'high_fever', 'hip_joint_pain', 'history_of_alcohol_consumption', 'increased_appetite',
    'indigestion', 'inflammatory_nails', 'internal_itching', 'irregular_sugar_level',
    'irritability', 'irritation_in_anus', 'joint_pain', 'knee_pain', 'lack_of_concentration',
    'lethargy', 'loss_of_appetite', 'loss_of_balance', 'loss_of_smell', 'malaise',
    'mild_fever', 'mood_swings', 'movement_stiffness', 'mucoid_sputum', 'muscle_pain',
    'muscle_wasting', 'muscle_weakness', 'nausea', 'neck_pain', 'nodal_skin_eruptions',
    'obesity', 'pain_behind_the_eyes', 'pain_during_bowel_movements', 'pain_in_anal_region',
    'painful_walking', 'palpitations', 'passage_of_gases', 'patches_in_throat',
    'phlegm', 'polyuria', 'prominent_veins_on_calf', 'puffy_face_and_eyes',
    'pus_filled_pimples', 'receiving_blood_transfusion', 'receiving_unsterile_injections',
    'red_sore_around_nose', 'red_spots_over_body', 'redness_of_eyes', 'restlessness',
    'runny_nose', 'rusty_sputum', 'scurring', 'shivering', 'silver_like_dusting',
    'sinus_pressure', 'skin_peeling', 'skin_rash', 'slurred_speech', 'small_dents_in_nails',
    'spinning_movements', 'spotting_urination', 'stiff_neck', 'stomach_bleeding',
    'stomach_pain', 'sunken_eyes', 'sweating', 'swelled_lymph_nodes', 'swelling_joints',
    'swelling_of_stomach', 'swollen_blood_vessels', 'swollen_extremeties', 'swollen_legs',
    'throat_irritation', 'toxic_look_(typhos)', 'ulcers_on_tongue', 'unsteadiness',
    'visual_disturbances', 'vomiting', 'watering_from_eyes', 'weakness_in_limbs',
    'weakness_of_one_body_side', 'weight_gain', 'weight_loss', 'yellow_crust_ooze',
    'yellow_urine', 'yellowing_of_eyes', 'yellowish_skin'
]

# Lazy loading the trained model
_model = None
def load_model():
    global _model
    if _model is None:
        model_path = os.path.join(BASE_PATH, "my_neural_network_model.h5")
        _model = keras.models.load_model(model_path)
    return _model

def predict_disease(user_symptoms):
    with open(os.path.join(BASE_PATH, "training_data.txt"), "r") as file:
        data = ast.literal_eval(file.read())

    all_symptoms = list(data.keys())
    input_vector = [0] * len(all_symptoms)

    for symptom in user_symptoms:
        if symptom in all_symptoms:
            input_vector[all_symptoms.index(symptom)] = 1
        else:
            logger.warning(f"Symptom '{symptom}' not found in training data.")

    model = load_model()
    input_array = np.array([input_vector])
    prediction = model.predict(input_array)[0]

    predicted_index = np.argmax(prediction)
    encoder = joblib.load(os.path.join(BASE_PATH, "label_encoder.joblib"))
    predicted_disease = encoder.inverse_transform([predicted_index])[0]

    return predicted_disease

def get_disease_information(disease):
    info = {
        'description': description[description['Disease'].str.lower() == disease.lower()]['Description'].values[0],
        'precautions': precautions[precautions['Disease'].str.lower() == disease.lower()]['Precaution'].values[0].split(','),
        'medications': medications[medications['Disease'].str.lower() == disease.lower()]['Medication'].values[0].split(','),
        'diets': diets[diets['Disease'].str.lower() == disease.lower()]['Diet'].values[0].split(','),
        'doctor': doc_vs_dis[doc_vs_dis['Disease'].str.lower() == disease.lower()]['Specialist'].values[0]
    }
    return info

def correct_symptom_name(input_symptom):
    match, score = process.extractOne(input_symptom, symptoms)
    return match if score > 60 else input_symptom
