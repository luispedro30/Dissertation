from flask import Flask, render_template, request, session
import pandas as pd
import pickle
import os
from os.path import join
from werkzeug.utils import secure_filename



UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size: 16MB

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

app.secret_key = 'This is your secret key to utilize session in Flask'

model_math_binary = pickle.load(open('Models/Maths/Binary/LinearRegression.pkl', 'rb'))
model_por_binary = pickle.load(open('Models/Portuguese/Binary/LinearRegression.pkl', 'rb'))
model_math_five_levels = pickle.load(open('Models/Maths/FiveLevels/LinearRegression.pkl', 'rb'))
model_por_five_levels = pickle.load(open('Models/Portuguese/FiveLevels/LinearRegression.pkl', 'rb'))
model_math_grade = pickle.load(open('Models/Maths/Grade/LinearRegression.pkl', 'rb'))
model_por_grade = pickle.load(open('Models/Portuguese/Grade/LinearRegression.pkl', 'rb'))
model_rfe_ensemble = pickle.load(open('Models/XGBoost.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def hello():
    return render_template('index.html')

@app.route('/DataAnalysis/Maths', methods=['GET', 'POST'])
def DataAnalysisMath():
    return render_template('DataAnalysisMath.html')

@app.route('/DataAnalysis/Portuguese', methods=['GET', 'POST'])
def DataAnalysisPort():
    return render_template('DataAnalysisPort.html')

@app.route('/Contacts', methods=['GET', 'POST'])
def contacts():
    return render_template('contacts.html')

def allowed_file(filename):
    # Specify the allowed file extensions
    ALLOWED_EXTENSIONS = {'csv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

"""
@app.route('/csv', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
        f = request.files.get('file')
        if f and allowed_file(f.filename):
            data_filename = secure_filename(f.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
            f.save(file_path)
            session['uploaded_data_file_path'] = file_path
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Get the first row of the DataFrame (excluding the header)
            data = df.iloc[0]
            
            # Convert the data to variables
            variables = {col: value for col, value in data.items()}
            
            # Make prediction using the model and variables
            # Note: You need to preprocess the data appropriately before making predictions
            # For example, if the model expects numerical inputs, convert the variables to numerical format
            
            # Assuming you have preprocessed the variables appropriately and stored them in X_pred
            X_pred = pd.DataFrame(data).transpose()  # Convert data to DataFrame and transpose it
            prediction = model_rfe_ensemble.predict(X_pred)
            
            # Optionally, you can store the prediction in the session
            session['prediction'] = prediction.tolist()

            print(session['prediction'])
            
            return render_template('CsvImport2.html', variables=variables, prediction=prediction)
    
    return render_template("CsvImport.html")


@app.route('/csv/show_data')
def showData():
    data_file_path = session.get('uploaded_data_file_path', None)
    if data_file_path:
        uploaded_df = pd.read_csv(data_file_path, encoding='unicode_escape')
        uploaded_df_html = uploaded_df.to_html()
        return render_template('show_csv_data.html', data_var=uploaded_df_html)
    return "No data file uploaded."

if __name__ == '__main__':
    app.run(debug=True)
"""
@app.route('/csv', methods=['GET', 'POST'])
def upload_and_show_data():
    if request.method == 'POST':
        # Handle file upload
        f = request.files.get('file')
        if f and allowed_file(f.filename):
            data_filename = secure_filename(f.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
            f.save(file_path)
            session['uploaded_data_file_path'] = file_path
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Get the first row of the DataFrame (excluding the header)
            data = df.iloc[0]
            
            # Convert the data to variables
            variables = {col: value for col, value in data.items()}
            
            # Make prediction using the model and variables
            # Note: You need to preprocess the data appropriately before making predictions
            # For example, if the model expects numerical inputs, convert the variables to numerical format
            
            # Assuming you have preprocessed the variables appropriately and stored them in X_pred
            X_pred = pd.DataFrame(data).transpose()  # Convert data to DataFrame and transpose it
            prediction = model_rfe_ensemble.predict(X_pred)
            
            # Optionally, you can store the prediction in the session
            session['prediction'] = prediction.tolist()

            print(session['prediction'])
            
            # Render the same HTML template with variables and prediction
            return render_template('CsvImport.html', variables=variables, prediction=prediction)
    
    # Handle GET request or failed POST request
    return render_template("CsvImport.html")


     

@app.route('/FiveLevel/Maths', methods = ['GET', 'POST'])
def FiveLevelsMaths():
    """Grabs the input values and uses them to make prediction between pass or fail"""
    msg = ""
    if request.method == "POST":
        school = request.form['school']
        if school == 'GP':
            school = 0.0
        else:
            school = 1.0
        sex = request.form['sex']
        if sex == 'F':
            sex = 0.0
        else:
            sex = 1.0
        age = request.form['age']
        address = request.form['address']
        if address == 'R':
            address = 0.0
        else:
            address = 1.0
        famsize = request.form['famsize']
        if famsize == 'LE3':
            famsize = 0.0
        else:
            famsize = 1.0
        Pstatus = request.form['Pstatus']
        if Pstatus == 'T':
            Pstatus = 0.0
        else:
            Pstatus = 1.0
        Medu = int(request.form['Medu'])
        Fedu = int(request.form['Fedu'])
        Mjob = request.form['Mjob']
        if Mjob == 'teacher':
            mjob_teacher = 1.0
            mjob_healthy = 0.0
            mjob_services = 0.0
            mjob_at_home = 0.0
            mjob_other = 0.0
        elif Mjob == 'health':
            mjob_teacher = 0.0
            mjob_healthy = 1.0
            mjob_services = 0.0
            mjob_at_home = 0.0
            mjob_other = 0.0
        elif Mjob == 'services':
            mjob_teacher = 0.0
            mjob_healthy = 0.0
            mjob_services = 1.0
            mjob_at_home = 0.0
            mjob_other = 0.0
        elif Mjob == 'at_home':
            mjob_teacher = 0.0
            mjob_healthy = 0.0
            mjob_services = 0.0
            mjob_at_home = 1.0
            mjob_other = 0.0
        else:
            mjob_teacher = 0.0
            mjob_healthy = 0.0
            mjob_services = 0.0
            mjob_at_home = 0.0
            mjob_other = 1.0
        Fjob = request.form['Fjob']
        if Fjob == 'teacher':
            fjob_teacher = 1.0
            fjob_healthy = 0.0
            fjob_services = 0.0
            fjob_at_home = 0.0
            fjob_other = 0.0
        elif Fjob == 'health':
            fjob_teacher = 0.0
            fjob_healthy = 1.0
            fjob_services = 0.0
            fjob_at_home = 0.0
            fjob_other = 0.0
        elif Fjob == 'services':
            fjob_teacher = 0.0
            fjob_healthy = 0.0
            fjob_services = 1.0
            fjob_at_home = 0.0
            fjob_other = 0.0
        elif Fjob == 'at_home':
            fjob_teacher = 0.0
            fjob_healthy = 0.0
            fjob_services = 0.0
            fjob_at_home = 1.0
            fjob_other = 0.0
        else:
            fjob_teacher = 0.0
            fjob_healthy = 0.0
            fjob_services = 0.0
            fjob_at_home = 0.0
            fjob_other = 1.0
        reason = request.form['reason']
        if reason == "home":
            reason_home = 1.0
            reason_reputation = 0.0
            reason_course = 0.0
            reason_other = 0.0
        elif reason == "reputation":
            reason_home = 0.0
            reason_reputation = 1.0
            reason_course = 0.0
            reason_other = 0.0
        elif reason == "course":
            reason_home = 0.0
            reason_reputation = 0.0
            reason_course = 1.0
            reason_other = 0.0
        else:
            reason_home = 0.0
            reason_reputation = 0.0
            reason_course = 0.0
            reason_other = 1.0
        guardian = request.form['guardian']
        if guardian == "mother":
            guardian_mother = 1.0
            guardian_father = 0.0
            guardian_other = 0.0
        elif guardian == "father":
            guardian_mother = 0.0
            guardian_father = 1.0
            guardian_other = 0.0
        else:
            guardian_mother = 0.0
            guardian_father = 0.0
            guardian_other = 1.0
        traveltime = int(request.form['traveltime'])
        studytime = int(request.form['studytime'])
        failures = int(request.form['failures'])
        schoolsup = request.form['schoolsup']
        if schoolsup == 'no':
            schoolsup = 0.0
        else:
            schoolsup = 1.0
        famsup = request.form['famsup']
        if famsup == 'no':
            famsup = 0.0
        else:
            famsup = 1.0
        paid = request.form['paid']
        if paid == 'no':
            paid = 0.0
        else:
            paid = 1.0
        activities = request.form['activities']
        if activities == 'no':
            activities = 0.0
        else:
            activities = 1.0
        nursery = request.form['nursery']
        if nursery == 'no':
            nursery = 0.0
        else:
            nursery = 1.0
        higher = request.form['higher']
        if higher == 'no':
            higher = 0.0
        else:
            higher = 1.0
        internet = request.form['internet']
        if internet == 'no':
            internet = 0.0
        else:
            internet = 1.0
        romantic = request.form['romantic']
        if romantic == 'no':
            romantic = 0.0
        else:
            romantic = 1.0
        famrel = int(request.form['famrel'])
        freetime = int(request.form['freetime'])
        goout = int(request.form['goout'])
        Dalc = int(request.form['Dalc'])
        Walc = int(request.form['Walc'])
        health = int(request.form['health'])
        absences = request.form['absences']
        G1 = int(request.form['G1'])
        if G1 <= 9:
            G1 = 5
        elif G1>= 10 and G1<=11:
            G1 = 4
        elif G1>= 12 and G1<=13:
            G1 = 3
        elif G1>= 14 and G1<=15:
            G1 = 2
        else:
            G1 = 1
        G2 = int(request.form['G2'])
        if G2 <= 9:
            G2 = 5
        elif G2>= 10 and G2<=11:
            G2 = 4
        elif G2>= 12 and G2<=13:
            G2 = 3
        elif G2>= 14 and G2<=15:
            G2 = 2
        else:
            G2 = 1

        prediction = model_math_five_levels.predict([[school, sex,address,famsize,Pstatus,
        mjob_healthy,mjob_other, mjob_services, mjob_teacher,
        fjob_healthy,fjob_other, fjob_services, fjob_teacher,
        reason_home,reason_other,reason_reputation,
        guardian_mother,guardian_other,
        schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,int(age),Medu,
        Fedu,traveltime,studytime,failures,famrel,freetime,goout,Dalc,Walc,health,int(absences),G1,G2]])
        print(prediction)
        print(prediction[0][0])
        if round(prediction[0][0]) == 1:
            msg = 'I - Very good'
        elif round(prediction[0][0]) == 2:
            msg = 'II - Good'
        elif round(prediction[0][0]) == 3:
            msg = 'III - Satisfatory'
        elif round(prediction[0][0]) == 4:
            msg = 'IV - Insatisfatory'
        else:
            msg = "V - Insufficient"
    return render_template('passFail.html', msg = msg)

@app.route('/FiveLevel/Portuguese', methods = ['GET', 'POST'])
def FiveLevelsPortuguese():
    """Grabs the input values and uses them to make prediction between pass or fail"""
    msg = ""
    if request.method == "POST":
        school = request.form['school']
        if school == 'GP':
            school = 0.0
        else:
            school = 1.0
        sex = request.form['sex']
        if sex == 'F':
            sex = 0.0
        else:
            sex = 1.0
        age = request.form['age']
        address = request.form['address']
        if address == 'R':
            address = 0.0
        else:
            address = 1.0
        famsize = request.form['famsize']
        if famsize == 'LE3':
            famsize = 0.0
        else:
            famsize = 1.0
        Pstatus = request.form['Pstatus']
        if Pstatus == 'T':
            Pstatus = 0.0
        else:
            Pstatus = 1.0
        Medu = int(request.form['Medu'])
        Fedu = int(request.form['Fedu'])
        Mjob = request.form['Mjob']
        if Mjob == 'teacher':
            mjob_teacher = 1.0
            mjob_healthy = 0.0
            mjob_services = 0.0
            mjob_at_home = 0.0
            mjob_other = 0.0
        elif Mjob == 'health':
            mjob_teacher = 0.0
            mjob_healthy = 1.0
            mjob_services = 0.0
            mjob_at_home = 0.0
            mjob_other = 0.0
        elif Mjob == 'services':
            mjob_teacher = 0.0
            mjob_healthy = 0.0
            mjob_services = 1.0
            mjob_at_home = 0.0
            mjob_other = 0.0
        elif Mjob == 'at_home':
            mjob_teacher = 0.0
            mjob_healthy = 0.0
            mjob_services = 0.0
            mjob_at_home = 1.0
            mjob_other = 0.0
        else:
            mjob_teacher = 0.0
            mjob_healthy = 0.0
            mjob_services = 0.0
            mjob_at_home = 0.0
            mjob_other = 1.0
        Fjob = request.form['Fjob']
        if Fjob == 'teacher':
            fjob_teacher = 1.0
            fjob_healthy = 0.0
            fjob_services = 0.0
            fjob_at_home = 0.0
            fjob_other = 0.0
        elif Fjob == 'health':
            fjob_teacher = 0.0
            fjob_healthy = 1.0
            fjob_services = 0.0
            fjob_at_home = 0.0
            fjob_other = 0.0
        elif Fjob == 'services':
            fjob_teacher = 0.0
            fjob_healthy = 0.0
            fjob_services = 1.0
            fjob_at_home = 0.0
            fjob_other = 0.0
        elif Fjob == 'at_home':
            fjob_teacher = 0.0
            fjob_healthy = 0.0
            fjob_services = 0.0
            fjob_at_home = 1.0
            fjob_other = 0.0
        else:
            fjob_teacher = 0.0
            fjob_healthy = 0.0
            fjob_services = 0.0
            fjob_at_home = 0.0
            fjob_other = 1.0
        reason = request.form['reason']
        if reason == "home":
            reason_home = 1.0
            reason_reputation = 0.0
            reason_course = 0.0
            reason_other = 0.0
        elif reason == "reputation":
            reason_home = 0.0
            reason_reputation = 1.0
            reason_course = 0.0
            reason_other = 0.0
        elif reason == "course":
            reason_home = 0.0
            reason_reputation = 0.0
            reason_course = 1.0
            reason_other = 0.0
        else:
            reason_home = 0.0
            reason_reputation = 0.0
            reason_course = 0.0
            reason_other = 1.0
        guardian = request.form['guardian']
        if guardian == "mother":
            guardian_mother = 1.0
            guardian_father = 0.0
            guardian_other = 0.0
        elif guardian == "father":
            guardian_mother = 0.0
            guardian_father = 1.0
            guardian_other = 0.0
        else:
            guardian_mother = 0.0
            guardian_father = 0.0
            guardian_other = 1.0
        traveltime = int(request.form['traveltime'])
        studytime = int(request.form['studytime'])
        failures = int(request.form['failures'])
        schoolsup = request.form['schoolsup']
        if schoolsup == 'no':
            schoolsup = 0.0
        else:
            schoolsup = 1.0
        famsup = request.form['famsup']
        if famsup == 'no':
            famsup = 0.0
        else:
            famsup = 1.0
        paid = request.form['paid']
        if paid == 'no':
            paid = 0.0
        else:
            paid = 1.0
        activities = request.form['activities']
        if activities == 'no':
            activities = 0.0
        else:
            activities = 1.0
        nursery = request.form['nursery']
        if nursery == 'no':
            nursery = 0.0
        else:
            nursery = 1.0
        higher = request.form['higher']
        if higher == 'no':
            higher = 0.0
        else:
            higher = 1.0
        internet = request.form['internet']
        if internet == 'no':
            internet = 0.0
        else:
            internet = 1.0
        romantic = request.form['romantic']
        if romantic == 'no':
            romantic = 0.0
        else:
            romantic = 1.0
        famrel = int(request.form['famrel'])
        freetime = int(request.form['freetime'])
        goout = int(request.form['goout'])
        Dalc = int(request.form['Dalc'])
        Walc = int(request.form['Walc'])
        health = int(request.form['health'])
        absences = request.form['absences']
        G1 = int(request.form['G1'])
        if G1 <= 9:
            G1 = 5
        elif G1>= 10 and G1<=11:
            G1 = 4
        elif G1>= 12 and G1<=13:
            G1 = 3
        elif G1>= 14 and G1<=15:
            G1 = 2
        else:
            G1 = 1
        G2 = int(request.form['G2'])
        if G2 <= 9:
            G2 = 5
        elif G2>= 10 and G2<=11:
            G2 = 4
        elif G2>= 12 and G2<=13:
            G2 = 3
        elif G2>= 14 and G2<=15:
            G2 = 2
        else:
            G2 = 1

        prediction = model_por_five_levels.predict([[school, sex,address,famsize,Pstatus,
        mjob_healthy,mjob_other, mjob_services, mjob_teacher,
        fjob_healthy,fjob_other, fjob_services, fjob_teacher,
        reason_home,reason_other,reason_reputation,
        guardian_mother,guardian_other,
        schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,int(age),Medu,
        Fedu,traveltime,studytime,failures,famrel,freetime,goout,Dalc,Walc,health,int(absences),G1,G2]])
        print(prediction)
        print(prediction[0][0])
        if round(prediction[0][0]) == 1:
            msg = 'I - Very good'
        elif round(prediction[0][0]) == 2:
            msg = 'II - Good'
        elif round(prediction[0][0]) == 3:
            msg = 'III - Satisfatory'
        elif round(prediction[0][0]) == 4:
            msg = 'IV - Insatisfatory'
        else:
            msg = "V - Insufficient"
    return render_template('passFail.html', msg = msg)

@app.route('/GradePrediction/Maths', methods = ['GET', 'POST'])
def GradeMaths():
    """Grabs the input values and uses them to make prediction between pass or fail"""
    msg = ""
    if request.method == "POST":
        school = request.form['school']
        if school == 'GP':
            school = 0.0
        else:
            school = 1.0
        sex = request.form['sex']
        if sex == 'F':
            sex = 0.0
        else:
            sex = 1.0
        age = request.form['age']
        address = request.form['address']
        if address == 'R':
            address = 0.0
        else:
            address = 1.0
        famsize = request.form['famsize']
        if famsize == 'LE3':
            famsize = 0.0
        else:
            famsize = 1.0
        Pstatus = request.form['Pstatus']
        if Pstatus == 'T':
            Pstatus = 0.0
        else:
            Pstatus = 1.0
        Medu = int(request.form['Medu'])
        Fedu = int(request.form['Fedu'])
        Mjob = request.form['Mjob']
        if Mjob == 'teacher':
            mjob_teacher = 1.0
            mjob_healthy = 0.0
            mjob_services = 0.0
            mjob_at_home = 0.0
            mjob_other = 0.0
        elif Mjob == 'health':
            mjob_teacher = 0.0
            mjob_healthy = 1.0
            mjob_services = 0.0
            mjob_at_home = 0.0
            mjob_other = 0.0
        elif Mjob == 'services':
            mjob_teacher = 0.0
            mjob_healthy = 0.0
            mjob_services = 1.0
            mjob_at_home = 0.0
            mjob_other = 0.0
        elif Mjob == 'at_home':
            mjob_teacher = 0.0
            mjob_healthy = 0.0
            mjob_services = 0.0
            mjob_at_home = 1.0
            mjob_other = 0.0
        else:
            mjob_teacher = 0.0
            mjob_healthy = 0.0
            mjob_services = 0.0
            mjob_at_home = 0.0
            mjob_other = 1.0
        Fjob = request.form['Fjob']
        if Fjob == 'teacher':
            fjob_teacher = 1.0
            fjob_healthy = 0.0
            fjob_services = 0.0
            fjob_at_home = 0.0
            fjob_other = 0.0
        elif Fjob == 'health':
            fjob_teacher = 0.0
            fjob_healthy = 1.0
            fjob_services = 0.0
            fjob_at_home = 0.0
            fjob_other = 0.0
        elif Fjob == 'services':
            fjob_teacher = 0.0
            fjob_healthy = 0.0
            fjob_services = 1.0
            fjob_at_home = 0.0
            fjob_other = 0.0
        elif Fjob == 'at_home':
            fjob_teacher = 0.0
            fjob_healthy = 0.0
            fjob_services = 0.0
            fjob_at_home = 1.0
            fjob_other = 0.0
        else:
            fjob_teacher = 0.0
            fjob_healthy = 0.0
            fjob_services = 0.0
            fjob_at_home = 0.0
            fjob_other = 1.0
        reason = request.form['reason']
        if reason == "home":
            reason_home = 1.0
            reason_reputation = 0.0
            reason_course = 0.0
            reason_other = 0.0
        elif reason == "reputation":
            reason_home = 0.0
            reason_reputation = 1.0
            reason_course = 0.0
            reason_other = 0.0
        elif reason == "course":
            reason_home = 0.0
            reason_reputation = 0.0
            reason_course = 1.0
            reason_other = 0.0
        else:
            reason_home = 0.0
            reason_reputation = 0.0
            reason_course = 0.0
            reason_other = 1.0
        guardian = request.form['guardian']
        if guardian == "mother":
            guardian_mother = 1.0
            guardian_father = 0.0
            guardian_other = 0.0
        elif guardian == "father":
            guardian_mother = 0.0
            guardian_father = 1.0
            guardian_other = 0.0
        else:
            guardian_mother = 0.0
            guardian_father = 0.0
            guardian_other = 1.0
        traveltime = int(request.form['traveltime'])
        studytime = int(request.form['studytime'])
        failures = int(request.form['failures'])
        schoolsup = request.form['schoolsup']
        if schoolsup == 'no':
            schoolsup = 0.0
        else:
            schoolsup = 1.0
        famsup = request.form['famsup']
        if famsup == 'no':
            famsup = 0.0
        else:
            famsup = 1.0
        paid = request.form['paid']
        if paid == 'no':
            paid = 0.0
        else:
            paid = 1.0
        activities = request.form['activities']
        if activities == 'no':
            activities = 0.0
        else:
            activities = 1.0
        nursery = request.form['nursery']
        if nursery == 'no':
            nursery = 0.0
        else:
            nursery = 1.0
        higher = request.form['higher']
        if higher == 'no':
            higher = 0.0
        else:
            higher = 1.0
        internet = request.form['internet']
        if internet == 'no':
            internet = 0.0
        else:
            internet = 1.0
        romantic = request.form['romantic']
        if romantic == 'no':
            romantic = 0.0
        else:
            romantic = 1.0
        famrel = int(request.form['famrel'])
        freetime = int(request.form['freetime'])
        goout = int(request.form['goout'])
        Dalc = int(request.form['Dalc'])
        Walc = int(request.form['Walc'])
        health = int(request.form['health'])
        absences = request.form['absences']
        G1 = int(request.form['G1'])
        G2 = int(request.form['G2'])

        prediction = model_math_grade.predict([[school, sex,address,famsize,Pstatus,
        mjob_healthy,mjob_other, mjob_services, mjob_teacher,
        fjob_healthy,fjob_other, fjob_services, fjob_teacher,
        reason_home,reason_other,reason_reputation,
        guardian_mother,guardian_other,
        schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,int(age),Medu,
        Fedu,traveltime,studytime,failures,famrel,freetime,goout,Dalc,Walc,health,int(absences),G1,G2]])
        if (prediction[0][0]) < 0:
            prediction[0][0] = 1
        msg = "Predicted grade: "+str(round(prediction[0][0]))
    return render_template('passFail.html',msg = msg)

@app.route('/GradePrediction/Portuguese', methods = ['GET', 'POST'])
def GradePortuguese():
    """Grabs the input values and uses them to make prediction between pass or fail"""
    msg = ""
    if request.method == "POST":
        school = request.form['school']
        if school == 'GP':
            school = 0.0
        else:
            school = 1.0
        sex = request.form['sex']
        if sex == 'F':
            sex = 0.0
        else:
            sex = 1.0
        age = request.form['age']
        address = request.form['address']
        if address == 'R':
            address = 0.0
        else:
            address = 1.0
        famsize = request.form['famsize']
        if famsize == 'LE3':
            famsize = 0.0
        else:
            famsize = 1.0
        Pstatus = request.form['Pstatus']
        if Pstatus == 'T':
            Pstatus = 0.0
        else:
            Pstatus = 1.0
        Medu = int(request.form['Medu'])
        Fedu = int(request.form['Fedu'])
        Mjob = request.form['Mjob']
        if Mjob == 'teacher':
            mjob_teacher = 1.0
            mjob_healthy = 0.0
            mjob_services = 0.0
            mjob_at_home = 0.0
            mjob_other = 0.0
        elif Mjob == 'health':
            mjob_teacher = 0.0
            mjob_healthy = 1.0
            mjob_services = 0.0
            mjob_at_home = 0.0
            mjob_other = 0.0
        elif Mjob == 'services':
            mjob_teacher = 0.0
            mjob_healthy = 0.0
            mjob_services = 1.0
            mjob_at_home = 0.0
            mjob_other = 0.0
        elif Mjob == 'at_home':
            mjob_teacher = 0.0
            mjob_healthy = 0.0
            mjob_services = 0.0
            mjob_at_home = 1.0
            mjob_other = 0.0
        else:
            mjob_teacher = 0.0
            mjob_healthy = 0.0
            mjob_services = 0.0
            mjob_at_home = 0.0
            mjob_other = 1.0
        Fjob = request.form['Fjob']
        if Fjob == 'teacher':
            fjob_teacher = 1.0
            fjob_healthy = 0.0
            fjob_services = 0.0
            fjob_at_home = 0.0
            fjob_other = 0.0
        elif Fjob == 'health':
            fjob_teacher = 0.0
            fjob_healthy = 1.0
            fjob_services = 0.0
            fjob_at_home = 0.0
            fjob_other = 0.0
        elif Fjob == 'services':
            fjob_teacher = 0.0
            fjob_healthy = 0.0
            fjob_services = 1.0
            fjob_at_home = 0.0
            fjob_other = 0.0
        elif Fjob == 'at_home':
            fjob_teacher = 0.0
            fjob_healthy = 0.0
            fjob_services = 0.0
            fjob_at_home = 1.0
            fjob_other = 0.0
        else:
            fjob_teacher = 0.0
            fjob_healthy = 0.0
            fjob_services = 0.0
            fjob_at_home = 0.0
            fjob_other = 1.0
        reason = request.form['reason']
        if reason == "home":
            reason_home = 1.0
            reason_reputation = 0.0
            reason_course = 0.0
            reason_other = 0.0
        elif reason == "reputation":
            reason_home = 0.0
            reason_reputation = 1.0
            reason_course = 0.0
            reason_other = 0.0
        elif reason == "course":
            reason_home = 0.0
            reason_reputation = 0.0
            reason_course = 1.0
            reason_other = 0.0
        else:
            reason_home = 0.0
            reason_reputation = 0.0
            reason_course = 0.0
            reason_other = 1.0
        guardian = request.form['guardian']
        if guardian == "mother":
            guardian_mother = 1.0
            guardian_father = 0.0
            guardian_other = 0.0
        elif guardian == "father":
            guardian_mother = 0.0
            guardian_father = 1.0
            guardian_other = 0.0
        else:
            guardian_mother = 0.0
            guardian_father = 0.0
            guardian_other = 1.0
        traveltime = int(request.form['traveltime'])
        studytime = int(request.form['studytime'])
        failures = int(request.form['failures'])
        schoolsup = request.form['schoolsup']
        if schoolsup == 'no':
            schoolsup = 0.0
        else:
            schoolsup = 1.0
        famsup = request.form['famsup']
        if famsup == 'no':
            famsup = 0.0
        else:
            famsup = 1.0
        paid = request.form['paid']
        if paid == 'no':
            paid = 0.0
        else:
            paid = 1.0
        activities = request.form['activities']
        if activities == 'no':
            activities = 0.0
        else:
            activities = 1.0
        nursery = request.form['nursery']
        if nursery == 'no':
            nursery = 0.0
        else:
            nursery = 1.0
        higher = request.form['higher']
        if higher == 'no':
            higher = 0.0
        else:
            higher = 1.0
        internet = request.form['internet']
        if internet == 'no':
            internet = 0.0
        else:
            internet = 1.0
        romantic = request.form['romantic']
        if romantic == 'no':
            romantic = 0.0
        else:
            romantic = 1.0
        famrel = int(request.form['famrel'])
        freetime = int(request.form['freetime'])
        goout = int(request.form['goout'])
        Dalc = int(request.form['Dalc'])
        Walc = int(request.form['Walc'])
        health = int(request.form['health'])
        absences = request.form['absences']
        G1 = int(request.form['G1'])
        G2 = int(request.form['G2'])

        prediction = model_por_grade.predict([[school, sex,address,famsize,Pstatus,
        mjob_healthy,mjob_other, mjob_services, mjob_teacher,
        fjob_healthy,fjob_other, fjob_services, fjob_teacher,
        reason_home,reason_other,reason_reputation,
        guardian_mother,guardian_other,
        schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,int(age),Medu,
        Fedu,traveltime,studytime,failures,famrel,freetime,goout,Dalc,Walc,health,int(absences),G1,G2]])
        print(prediction)
        msg = "Predicted grade: "+str(round(prediction[0][0]))
    return render_template('passFail.html',msg = msg)

@app.route('/PassOrFail/Portuguese',methods=['GET','POST'])
def passFailPortuguese():
    """Grabs the input values and uses them to make prediction between pass or fail"""
    msg = ""
    if request.method == "POST":
        school = request.form['school']
        if school == 'GP':
            school = 0.0
        else:
            school = 1.0
        sex = request.form['sex']
        if sex == 'F':
            sex = 0.0
        else:
            sex = 1.0
        age = request.form['age']
        address = request.form['address']
        if address == 'R':
            address = 0.0
        else:
            address = 1.0
        famsize = request.form['famsize']
        if famsize == 'LE3':
            famsize = 0.0
        else:
            famsize = 1.0
        Pstatus = request.form['Pstatus']
        if Pstatus == 'T':
            Pstatus = 0.0
        else:
            Pstatus = 1.0
        Medu = int(request.form['Medu'])
        Fedu = int(request.form['Fedu'])
        Mjob = request.form['Mjob']
        if Mjob == 'teacher':
            mjob_teacher = 1.0
            mjob_healthy = 0.0
            mjob_services = 0.0
            mjob_at_home = 0.0
            mjob_other = 0.0
        elif Mjob == 'health':
            mjob_teacher = 0.0
            mjob_healthy = 1.0
            mjob_services = 0.0
            mjob_at_home = 0.0
            mjob_other = 0.0
        elif Mjob == 'services':
            mjob_teacher = 0.0
            mjob_healthy = 0.0
            mjob_services = 1.0
            mjob_at_home = 0.0
            mjob_other = 0.0
        elif Mjob == 'at_home':
            mjob_teacher = 0.0
            mjob_healthy = 0.0
            mjob_services = 0.0
            mjob_at_home = 1.0
            mjob_other = 0.0
        else:
            mjob_teacher = 0.0
            mjob_healthy = 0.0
            mjob_services = 0.0
            mjob_at_home = 0.0
            mjob_other = 1.0
        Fjob = request.form['Fjob']
        if Fjob == 'teacher':
            fjob_teacher = 1.0
            fjob_healthy = 0.0
            fjob_services = 0.0
            fjob_at_home = 0.0
            fjob_other = 0.0
        elif Fjob == 'health':
            fjob_teacher = 0.0
            fjob_healthy = 1.0
            fjob_services = 0.0
            fjob_at_home = 0.0
            fjob_other = 0.0
        elif Fjob == 'services':
            fjob_teacher = 0.0
            fjob_healthy = 0.0
            fjob_services = 1.0
            fjob_at_home = 0.0
            fjob_other = 0.0
        elif Fjob == 'at_home':
            fjob_teacher = 0.0
            fjob_healthy = 0.0
            fjob_services = 0.0
            fjob_at_home = 1.0
            fjob_other = 0.0
        else:
            fjob_teacher = 0.0
            fjob_healthy = 0.0
            fjob_services = 0.0
            fjob_at_home = 0.0
            fjob_other = 1.0
        reason = request.form['reason']
        if reason == "home":
            reason_home = 1.0
            reason_reputation = 0.0
            reason_course = 0.0
            reason_other = 0.0
        elif reason == "reputation":
            reason_home = 0.0
            reason_reputation = 1.0
            reason_course = 0.0
            reason_other = 0.0
        elif reason == "course":
            reason_home = 0.0
            reason_reputation = 0.0
            reason_course = 1.0
            reason_other = 0.0
        else:
            reason_home = 0.0
            reason_reputation = 0.0
            reason_course = 0.0
            reason_other = 1.0
        guardian = request.form['guardian']
        if guardian == "mother":
            guardian_mother = 1.0
            guardian_father = 0.0
            guardian_other = 0.0
        elif guardian == "father":
            guardian_mother = 0.0
            guardian_father = 1.0
            guardian_other = 0.0
        else:
            guardian_mother = 0.0
            guardian_father = 0.0
            guardian_other = 1.0
        traveltime = int(request.form['traveltime'])
        studytime = int(request.form['studytime'])
        failures = int(request.form['failures'])
        schoolsup = request.form['schoolsup']
        if schoolsup == 'no':
            schoolsup = 0.0
        else:
            schoolsup = 1.0
        famsup = request.form['famsup']
        if famsup == 'no':
            famsup = 0.0
        else:
            famsup = 1.0
        paid = request.form['paid']
        if paid == 'no':
            paid = 0.0
        else:
            paid = 1.0
        activities = request.form['activities']
        if activities == 'no':
            activities = 0.0
        else:
            activities = 1.0
        nursery = request.form['nursery']
        if nursery == 'no':
            nursery = 0.0
        else:
            nursery = 1.0
        higher = request.form['higher']
        if higher == 'no':
            higher = 0.0
        else:
            higher = 1.0
        internet = request.form['internet']
        if internet == 'no':
            internet = 0.0
        else:
            internet = 1.0
        romantic = request.form['romantic']
        if romantic == 'no':
            romantic = 0.0
        else:
            romantic = 1.0
        famrel = int(request.form['famrel'])
        freetime = int(request.form['freetime'])
        goout = int(request.form['goout'])
        Dalc = int(request.form['Dalc'])
        Walc = int(request.form['Walc'])
        health = int(request.form['health'])
        absences = request.form['absences']
        G1 = float(request.form['G1'])
        if G1 >= 10:
            G1 = 1.0
        else:
            G1 = 0.0
        G2 = int(request.form['G2'])
        if G2 >= 10:
            G2 = 1.0
        else:
            G2 = 0.0
        prediction = model_por_binary.predict([[school, sex,address,famsize,Pstatus,
        mjob_healthy,mjob_other, mjob_services, mjob_teacher,
        fjob_healthy,fjob_services, fjob_teacher,
        reason_home,reason_other,reason_reputation,
        guardian_mother,guardian_other,
        schoolsup,famsup,paid,activities,nursery,internet,romantic,Medu,
        Fedu,traveltime,studytime,failures,Dalc,Walc,health,int(absences),G1,G2]])
        print((prediction))
        if round(prediction[0][0]) == 1:
            msg = "You are approved"
        else:
            msg = "You are repproved"
    return render_template('passFail.html',msg = msg)

@app.route('/PassOrFail/Maths',methods=['GET','POST'])
def passFailMaths():
    """Grabs the input values and uses them to make prediction between pass or fail"""
    msg = ""
    if request.method == "POST":
        school = request.form['school']
        if school == 'GP':
            school = 0.0
        else:
            school = 1.0
        sex = request.form['sex']
        if sex == 'F':
            sex = 0.0
        else:
            sex = 1.0
        age = request.form['age']
        address = request.form['address']
        if address == 'R':
            address = 0.0
        else:
            address = 1.0
        famsize = request.form['famsize']
        if famsize == 'LE3':
            famsize = 0.0
        else:
            famsize = 1.0
        Pstatus = request.form['Pstatus']
        if Pstatus == 'T':
            Pstatus = 0.0
        else:
            Pstatus = 1.0
        Medu = int(request.form['Medu'])
        Fedu = int(request.form['Fedu'])
        Mjob = request.form['Mjob']
        if Mjob == 'teacher':
            mjob_teacher = 1.0
            mjob_healthy = 0.0
            mjob_services = 0.0
            mjob_at_home = 0.0
            mjob_other = 0.0
        elif Mjob == 'health':
            mjob_teacher = 0.0
            mjob_healthy = 1.0
            mjob_services = 0.0
            mjob_at_home = 0.0
            mjob_other = 0.0
        elif Mjob == 'services':
            mjob_teacher = 0.0
            mjob_healthy = 0.0
            mjob_services = 1.0
            mjob_at_home = 0.0
            mjob_other = 0.0
        elif Mjob == 'at_home':
            mjob_teacher = 0.0
            mjob_healthy = 0.0
            mjob_services = 0.0
            mjob_at_home = 1.0
            mjob_other = 0.0
        else:
            mjob_teacher = 0.0
            mjob_healthy = 0.0
            mjob_services = 0.0
            mjob_at_home = 0.0
            mjob_other = 1.0
        Fjob = request.form['Fjob']
        if Fjob == 'teacher':
            fjob_teacher = 1.0
            fjob_healthy = 0.0
            fjob_services = 0.0
            fjob_at_home = 0.0
            fjob_other = 0.0
        elif Fjob == 'health':
            fjob_teacher = 0.0
            fjob_healthy = 1.0
            fjob_services = 0.0
            fjob_at_home = 0.0
            fjob_other = 0.0
        elif Fjob == 'services':
            fjob_teacher = 0.0
            fjob_healthy = 0.0
            fjob_services = 1.0
            fjob_at_home = 0.0
            fjob_other = 0.0
        elif Fjob == 'at_home':
            fjob_teacher = 0.0
            fjob_healthy = 0.0
            fjob_services = 0.0
            fjob_at_home = 1.0
            fjob_other = 0.0
        else:
            fjob_teacher = 0.0
            fjob_healthy = 0.0
            fjob_services = 0.0
            fjob_at_home = 0.0
            fjob_other = 1.0
        reason = request.form['reason']
        if reason == "home":
            reason_home = 1.0
            reason_reputation = 0.0
            reason_course = 0.0
            reason_other = 0.0
        elif reason == "reputation":
            reason_home = 0.0
            reason_reputation = 1.0
            reason_course = 0.0
            reason_other = 0.0
        elif reason == "course":
            reason_home = 0.0
            reason_reputation = 0.0
            reason_course = 1.0
            reason_other = 0.0
        else:
            reason_home = 0.0
            reason_reputation = 0.0
            reason_course = 0.0
            reason_other = 1.0
        guardian = request.form['guardian']
        if guardian == "mother":
            guardian_mother = 1.0
            guardian_father = 0.0
            guardian_other = 0.0
        elif guardian == "father":
            guardian_mother = 0.0
            guardian_father = 1.0
            guardian_other = 0.0
        else:
            guardian_mother = 0.0
            guardian_father = 0.0
            guardian_other = 1.0
        traveltime = int(request.form['traveltime'])
        studytime = int(request.form['studytime'])
        failures = int(request.form['failures'])
        schoolsup = request.form['schoolsup']
        if schoolsup == 'no':
            schoolsup = 0.0
        else:
            schoolsup = 1.0
        famsup = request.form['famsup']
        if famsup == 'no':
            famsup = 0.0
        else:
            famsup = 1.0
        paid = request.form['paid']
        if paid == 'no':
            paid = 0.0
        else:
            paid = 1.0
        activities = request.form['activities']
        if activities == 'no':
            activities = 0.0
        else:
            activities = 1.0
        nursery = request.form['nursery']
        if nursery == 'no':
            nursery = 0.0
        else:
            nursery = 1.0
        higher = request.form['higher']
        if higher == 'no':
            higher = 0.0
        else:
            higher = 1.0
        internet = request.form['internet']
        if internet == 'no':
            internet = 0.0
        else:
            internet = 1.0
        romantic = request.form['romantic']
        if romantic == 'no':
            romantic = 0.0
        else:
            romantic = 1.0
        famrel = int(request.form['famrel'])
        freetime = int(request.form['freetime'])
        goout = int(request.form['goout'])
        Dalc = int(request.form['Dalc'])
        Walc = int(request.form['Walc'])
        health = int(request.form['health'])
        absences = request.form['absences']
        G1 = float(request.form['G1'])
        if G1 >= 10:
            G1 = 1.0
        else:
            G1 = 0.0
        G2 = int(request.form['G2'])
        if G2 >= 10:
            G2 = 1.0
        else:
            G2 = 0.0
        prediction = model_math_binary.predict([[school, sex,address,famsize,Pstatus,
        mjob_healthy,mjob_other, mjob_services, mjob_teacher,
        fjob_healthy,fjob_services, fjob_teacher,
        reason_home,reason_other,reason_reputation,
        guardian_mother,guardian_other,
        schoolsup,famsup,paid,activities,nursery,internet,romantic,
        Fedu,traveltime,studytime,failures,Dalc,Walc,health,int(absences),G1,G2]])
        print((prediction))
        if round(prediction[0][0]) == 1:
            msg = "You are approved"
        else:
            msg = "You are repproved"
    return render_template('passFail.html',msg = msg)



    return render_template('gradePrediction.html')
if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = '8080')