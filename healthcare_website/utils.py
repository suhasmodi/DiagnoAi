# utils.py
import pandas as pd
from fuzzywuzzy import process
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder



symptoms=['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety',
          'cold_hands_and_feets','mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat',
          'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 
          'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 
          'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 
          'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']




import os
import pandas as pd
from django.conf import settings

dataset_dir = os.path.join(settings.BASE_DIR, "healthcare_website", "dataset")

# Load datasets using absolute paths
description = pd.read_csv(os.path.join(dataset_dir, "description.csv"))
precautions = pd.read_csv(os.path.join(dataset_dir, "precautions_df.csv"))
medications = pd.read_csv(os.path.join(dataset_dir, "medications.csv"))
diets = pd.read_csv(os.path.join(dataset_dir, "diets.csv"))
doc_vs_dis = pd.read_csv(os.path.join(dataset_dir, "Doctor_Versus_Disease.csv"), encoding='latin1')
workout = pd.read_csv(os.path.join(dataset_dir, "workout_df.csv"))


precautions.drop(columns='Unnamed: 0',inplace=True)
most_frequent = precautions['Precaution_3'].mode()[0]
precautions['Precaution_3'] = precautions['Precaution_3'].fillna(most_frequent)

most_frequent = precautions['Precaution_4'].mode()[0]
precautions['Precaution_4'] = precautions['Precaution_4'].fillna(most_frequent)
workout.drop(columns=['Unnamed: 0','Unnamed: 0.1'],inplace=True)
doc_vs_dis['Allergist'] = doc_vs_dis['Allergist'].replace({'Gastroenterologist\xa0':'Gastroenterologist'})


import os
import numpy as np
import pandas as pd
import joblib
import ast
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import process
from django.conf import settings
import tensorflow as tf
from tensorflow import keras

# Define dataset directory (adjust as needed)
dataset_dir = os.path.join(settings.BASE_DIR, "healthcare_website", "dataset")

# Load CSV files (assumes the CSV filenames below exist in dataset_dir)
description = pd.read_csv(os.path.join(dataset_dir, "description.csv"))
precautions = pd.read_csv(os.path.join(dataset_dir, "precautions_df.csv"))
medications = pd.read_csv(os.path.join(dataset_dir, "medications.csv"))
diets = pd.read_csv(os.path.join(dataset_dir, "diets.csv"))
doc_vs_dis = pd.read_csv(os.path.join(dataset_dir, "Doctor_Versus_Disease.csv"), encoding='latin1')
# 'workout' will be loaded later
workout = pd.read_csv(os.path.join(dataset_dir, "workout_df.csv"))

def get_disease_information(predicted_dis):
    global workout, description, precautions, medications, diets, doc_vs_dis

    # Load workout recommendations if not already loaded
    if not isinstance(workout, pd.DataFrame):
        workout = pd.read_csv(os.path.join(dataset_dir, "workout_df.csv"))

    # Extract disease description
    disease_description_series = description.loc[description['Disease'] == predicted_dis, 'Description']
    disease_description = (
        disease_description_series.iloc[0]
        if not disease_description_series.empty
        else "No description available"
    )

    # Extract precautions (assuming 4 precaution columns)
    disease_precautions_df = precautions.loc[
        precautions['Disease'] == predicted_dis,
        ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
    ]
    disease_precautions = (
        disease_precautions_df.values.tolist()
        if not disease_precautions_df.empty
        else [["No precautions available"]]
    )

    # Extract medications as a Series
    disease_medications_series = medications.loc[medications['Disease'] == predicted_dis, 'Medication']
    disease_medications = (
        disease_medications_series.tolist()
        if not disease_medications_series.empty
        else ["No medication information available"]
    )
    # Convert string representation of list into an actual list if needed
    if disease_medications and isinstance(disease_medications[0], str) and disease_medications[0].startswith('['):
        try:
            meds = ast.literal_eval(disease_medications[0])
            if isinstance(meds, list):
                disease_medications = meds
        except Exception as e:
            pass

    # Extract diet as a Series
    disease_diet_series = diets.loc[diets['Disease'] == predicted_dis, 'Diet']
    disease_diet = (
        disease_diet_series.tolist()
        if not disease_diet_series.empty
        else ["No diet information available"]
    )
    # Convert string representation to a list if needed
    if disease_diet and isinstance(disease_diet[0], str) and disease_diet[0].startswith('['):
        try:
            d_list = ast.literal_eval(disease_diet[0])
            if isinstance(d_list, list):
                disease_diet = d_list
        except Exception as e:
            pass

    # Extract workout recommendations as a Series
    disease_workout_series = workout.loc[workout['disease'] == predicted_dis, 'workout']
    disease_workout = (
        disease_workout_series.tolist()
        if not disease_workout_series.empty
        else ["No workout information available"]
    )

    # Extract recommended specialists as a Series
    doc_dis_series = doc_vs_dis.loc[doc_vs_dis['Drug Reaction'] == predicted_dis, 'Allergist']
    doc_dis = (
        doc_dis_series.tolist()
        if not doc_dis_series.empty
        else ["No specialist available"]
    )

    return (
        disease_description,
        disease_precautions,
        disease_medications,
        disease_diet,
        disease_workout,
        doc_dis
    )

def correct_spelling(symptom):
    """Uses fuzzy matching to return a corrected symptom if the confidence is high enough."""
    # Use the global 'symptoms' list as reference
    if not symptoms:
        return symptom
    match = process.extractOne(symptom, symptoms)
    if match and match[1] >= 80:
        return match[0]
    else:
        return symptom

def predict_disease(user_input):
    # Ensure all feature names match training features
    all_features = symptoms

    # Create a DataFrame with all features set to 0
    input_data = pd.DataFrame(np.zeros((1, len(all_features))), columns=all_features)

    # For each symptom in the input, set its value to 1 if it exists in the features
    for symptom in user_input:
        if symptom in input_data.columns:
            input_data[symptom] = 1
        else:
            print(f"Warning: Symptom '{symptom}' not found in the training data. Ignoring it.")

    # Load the saved model
    model_dir = os.path.join(settings.BASE_DIR, "healthcare_website")
    # model_path = os.path.join(model_dir, "RandomForest_model.pkl")
    model_path = os.path.join(model_dir, "my_neural_network_model.h5")
    model =  keras.models.load_model(model_path)
    # model=joblib.load(model_path)

    # Make a prediction (assuming model.predict returns probabilities or similar)
    prediction_probabilities = model.predict(input_data)

    # Get the index of the highest probability disease
    predicted_index = np.argmax(prediction_probabilities)

    # Convert the predicted index to a disease name using a LabelEncoder
    diseases = [
        'Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
        'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
        'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
        'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
        'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
        'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
        'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
        'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
        'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
        'Osteoarthristis', 'Arthritis',
        '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
        'Urinary tract infection', 'Psoriasis', 'Impetigo'
    ]
    le = LabelEncoder()
    le.fit(diseases)
    predicted_disease = le.inverse_transform([predicted_index])[0]

    return str(predicted_disease)


# x=predict_disease(['itching', 'skin_rash', 'nodal_skin_eruptions'])
# print(get_disease_information(x))