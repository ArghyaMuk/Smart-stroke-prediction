from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import pickle
app = Flask(__name__)
model = pickle.load(open('model_RandomForest.pkl', 'rb'))
scaler = pickle.load(open('StandarScaler.pkl', 'rb'))
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=['POST'])
def predict():
    gender=(request.form['gender'])
    age=int(request.form['age'])
    hypertension=(request.form['hypertension'])
    heart_disease =(request.form['heart_disease'])
    ever_married = (request.form['ever_married'])
    work_type = (request.form['work_type'])
    Residence_type = (request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = (request.form['smoking_status'])

    if gender=='male' or gender=='Male' or gender=='Male' or gender=='male':
        gender=1
    elif gender == 'female' or gender == 'Female' or gender == 'female' or gender == 'female':
        gender=0
    else:
        gender = 2

    if hypertension  == 'yes' or hypertension == 'YES':
        hypertension = 1
    else:
        hypertension=0

    if heart_disease  == 'yes' or heart_disease == 'YES':
        heart_disease =1
    else:
        heart_disease=0

    if work_type=='private' or work_type=='PRIVATE':
        work_type=2
    elif work_type=='self_employed' or work_type=='SELF_EMPLOYED':
        work_type=3
    elif work_type == 'goverment_job' or work_type=='GOVERMENT_JOB':
        work_type=0
    elif work_type== 'children' or work_type=='CHILDREN':
        work_type=4
    else:
        work_type=1

    if Residence_type=='URBAN' or Residence_type=='urban':
        Residence_type = 1
    elif Residence_type=='RURAL' or Residence_type=='rural' :
        Residence_type=0

    if smoking_status=='formerly somked' or smoking_status =='FORMERLY SMOKED' :
        smoking_status=1
    elif smoking_status=='never smoke' or smoking_status =='NEVER SMOKED' :
        smoking_status=2
    elif smoking_status=='somkes' or smoking_status =='SMOKES' :
        smoking_status=3
    elif smoking_status=='unknown' or smoking_status =='UNKNOWN' :
        smoking_status=0

    if ever_married=='yes' or ever_married=='YES':
        ever_married=1
    elif ever_married=='no' or ever_married=='NO':
        ever_married=0

    x=np.array([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,
                avg_glucose_level,bmi,smoking_status]).reshape(1,-1)
    
    x=scaler.transform(x)

    Y_pred = model.predict(x)

    # for No Stroke Risk
    if Y_pred==0:
        return render_template('no_stroke.html')
    else:
        return render_template('stroke.html')


if __name__ == '__main__':
    app.debug = True
    app.run()