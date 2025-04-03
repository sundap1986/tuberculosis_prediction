import gradio as gr
import requests

def predict_tb(Patient_ID, Age, Gender, Chest_Pain, Cough_Severity, Breathlessness, Fatigue, Weight_Loss, Fever, Night_Sweats, Sputum_Production, Blood_in_Sputum, Smoking_History, Previous_TB_History):
    url = "http://127.0.0.1:8000/predict"
    data = {
        "Patient_ID": Patient_ID,
        "Age": Age,
        "Gender": Gender,
        "Chest_Pain": Chest_Pain,
        "Cough_Severity": Cough_Severity,
        "Breathlessness": Breathlessness,
        "Fatigue": Fatigue,
        "Weight_Loss": Weight_Loss,
        "Fever": Fever,
        "Night_Sweats": Night_Sweats,
        "Sputum_Production": Sputum_Production,
        "Blood_in_Sputum": Blood_in_Sputum,
        "Smoking_History": Smoking_History,
        "Previous_TB_History": Previous_TB_History
    }
    
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()["prediction"]
    else:
        return "Error: Failed to get prediction!"

gr.Interface(
    fn=predict_tb,
    inputs=[
        "text",  # Patient_ID
        "number",  # Age
        gr.Radio(["Male", "Female"]),  # Gender
        gr.Radio(["Yes", "No"]),  # Chest_Pain
        "number",  # Cough_Severity
        "number",  # Breathlessness
        "number",  # Fatigue
        "number",  # Weight_Loss
        gr.Radio(["Mild", "Moderate", "High"]),  # Fever
        gr.Radio(["Yes", "No"]),  # Night_Sweats
        gr.Radio(["Low", "Medium", "High"]),  # Sputum_Production
        gr.Radio(["Yes", "No"]),  # Blood_in_Sputum
        gr.Radio(["Never", "Former", "Current"]),  # Smoking_History
        gr.Radio(["Yes", "No"])  # Previous_TB_History
    ],
    outputs="text"
).launch()
