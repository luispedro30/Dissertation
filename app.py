from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import pickle
import os
import joblib
from os.path import join
from werkzeug.utils import secure_filename

app = Flask(__name__)


UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSIONS = {'csv'}

app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://admin:postgres123@localhost:5433/ml"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key_here'  # Secret key for session management
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size: 16MB

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Specify the login route
login_manager.login_message = 'Please log in to access this page.'

# Define the User model with a custom table name
class User(UserMixin, db.Model):
    __tablename__ = 'users'  # Specify the custom table name

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(500), unique=True, nullable=False)
    email = db.Column(db.String(500), unique=True, nullable=False)
    password_hash = db.Column(db.String(500), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


# User loader function for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, user_id)


# Registration route
@app.route('/register', methods=['POST'])
def register_user():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    # Check if username or email already exists
    existing_user = db.session.query(User).filter_by(username=username).first()
    existing_email = db.session.query(User).filter_by(email=email).first()

    if existing_user:
        return make_response(jsonify({'error': 'Username already exists'}), 400)
    if existing_email:
        return make_response(jsonify({'error': 'Email already exists'}), 400)

    # Hash the password before storing it
    hashed_password = generate_password_hash(password)

    # Create a new user instance
    new_user = User(username=username, email=email, password_hash=hashed_password)

    # Add the new user to the database
    db.session.add(new_user)
    db.session.commit()

    return make_response(jsonify({'message': 'User registered successfully'}), 201)

# Login route
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    user = db.session.query(User).filter_by(username=username).first()



    if user and user.check_password(password):
        login_user(user)  # Log in the user
        return jsonify({'success': True, 'message': 'Login successful'})
    else:
        return jsonify({'success': False, 'message': 'Invalid username or password'})



@app.route('/', methods=['GET', 'POST'])
def hello():
    return render_template('registerlogin.html')

@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    return render_template('index.html')

@app.route('/DataAnalysis/Geral', methods=['GET', 'POST'])
def DataAnalysisMath():
    return render_template('DataAnalysisGeral.html')

@app.route('/DataAnalysis/Distribution', methods=['GET', 'POST'])
def DataAnalysisPort():
    return render_template('DataAnalysisDistribution.html')

@app.route('/Contacts', methods=['GET', 'POST'])
def contacts():
    return render_template('contacts.html')

def allowed_file(filename):
    # Specify the allowed file extensions
    ALLOWED_EXTENSIONS = {'csv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/AllDataset/AllDatasetNormalized', methods=['GET', 'POST'])
def upload_and_show_data():
    parameters = None  # Default value for parameters
    if request.method == 'POST':
        # Handle file upload
        f = request.files.get('file')
        if f and allowed_file(f.filename):
            data_filename = secure_filename(f.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
            f.save(file_path)
            session['uploaded_data_file_path'] = file_path
            model = None
            selected_model = request.form.get('model')

            print(selected_model)

            if selected_model == 'svm':
                model = joblib.load('Models/AllDatasetNormalized/SVM.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetNormalized/SVM_feature_names.csv")
                print("aqui")
            elif selected_model == 'nb':
                model = joblib.load('Models/AllDatasetNormalized/Naive Bayes.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetNormalized/Naive Bayes_feature_names.csv")
            elif selected_model == 'knn':
                model = joblib.load('Models/AllDatasetNormalized/KNN.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetNormalized/kNN_feature_names.csv")
            elif selected_model == 'adaboost':
                model = joblib.load('Models/AllDatasetNormalized/AdaBoost.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetNormalized/AdaBoost_feature_names.csv")
            elif selected_model == 'dt':
                model = joblib.load('Models/AllDatasetNormalized/Decision Tree.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetNormalized/Decision Tree_feature_names.csv")
            elif selected_model == 'xgboost':
                model = joblib.load('Models/AllDatasetNormalized/XGBoost.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetNormalized/XGBoost_feature_names.csv")
            elif selected_model == 'rf':
                model = joblib.load('Models/AllDatasetNormalized/Random Forest.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetNormalized/Random Forest_feature_names.csv")
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Get the feature names from the model
            # Extract the feature names from the DataFrame

            print(feature_names)
            selected_features = feature_names['Feature Names'].tolist()
            print(selected_features)
            #selected_features = model.get_booster().feature_names
            
            if hasattr(model, 'get_params'):
                parameters = model.get_params()
                print("parametros")
            else:
                parameters = None  # or any other way to handle this case

            # Extract data for selected features from the first row of the DataFrame
            data = df.loc[0, selected_features] 
            
            # Convert the data to variables
            variables = {col: value for col, value in data.items()}
            
            # Make prediction using the model and variables
            # Note: You need to preprocess the data appropriately before making predictions
            # For example, if the model expects numerical inputs, convert the variables to numerical format
            
            # Assuming you have preprocessed the variables appropriately and stored them in X_pred
            X_pred = pd.DataFrame(data).transpose()  # Convert data to DataFrame and transpose it
            prediction = model.predict(X_pred)
            
            # Optionally, you can store the prediction in the session
            session['prediction'] = prediction.tolist()

            print(session['prediction'])
            print(prediction)

            # Render the HTML template with variables, prediction, and model parameters
            return render_template('CsvImportAllDatasetNormalized.html', variables=variables, prediction=prediction, parameters=parameters, selected_model=selected_model)
    
    # Handle GET request or failed POST request
    return render_template("CsvImportAllDatasetNormalized.html", parameters=parameters)

@app.route('/AllDataset/AllDatasetNormalizedCorrelated', methods=['GET', 'POST'])
def upload_and_show_data_normalized_correlated():
    parameters = None  # Default value for parameters
    if request.method == 'POST':
        # Handle file upload
        f = request.files.get('file')
        if f and allowed_file(f.filename):
            data_filename = secure_filename(f.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
            f.save(file_path)
            session['uploaded_data_file_path'] = file_path
            model = None
            selected_model = request.form.get('model')

            print(selected_model)

            if selected_model == 'svm':
                model = joblib.load('Models/AllDatasetNormalizedCorrelated/SVM.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetNormalizedCorrelated/SVM_feature_names.csv")
                print("aqui")
            elif selected_model == 'nb':
                model = joblib.load('Models/AllDatasetNormalizedCorrelated/Naive Bayes.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetNormalizedCorrelated/Naive Bayes_feature_names.csv")
            elif selected_model == 'knn':
                model = joblib.load('Models/AllDatasetNormalizedCorrelated/KNN.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetNormalizedCorrelated/kNN_feature_names.csv")
            elif selected_model == 'adaboost':
                model = joblib.load('Models/AllDatasetNormalizedCorrelated/AdaBoost.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetNormalizedCorrelated/AdaBoost_feature_names.csv")
            elif selected_model == 'dt':
                model = joblib.load('Models/AllDatasetNormalizedCorrelated/Decision Tree.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetNormalizedCorrelated/Decision Tree_feature_names.csv")
            elif selected_model == 'xgboost':
                model = joblib.load('Models/AllDatasetNormalizedCorrelated/XGBoost.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetNormalizedCorrelated/XGBoost_feature_names.csv")
            elif selected_model == 'rf':
                model = joblib.load('Models/AllDatasetNormalizedCorrelated/Random Forest.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetNormalizedCorrelated/Random Forest_feature_names.csv")
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Get the feature names from the model
            # Extract the feature names from the DataFrame

            print(feature_names)
            selected_features = feature_names['Feature Names'].tolist()
            print(selected_features)
            #selected_features = model.get_booster().feature_names
            
            if hasattr(model, 'get_params'):
                parameters = model.get_params()
                print("parametros")
            else:
                parameters = None  # or any other way to handle this case

            # Extract data for selected features from the first row of the DataFrame
            data = df.loc[0, selected_features] 
            print(data)
            
            # Convert the data to variables
            variables = {col: value for col, value in data.items()}
            
            # Make prediction using the model and variables
            # Note: You need to preprocess the data appropriately before making predictions
            # For example, if the model expects numerical inputs, convert the variables to numerical format
            
            # Assuming you have preprocessed the variables appropriately and stored them in X_pred
            X_pred = pd.DataFrame(data).transpose()  # Convert data to DataFrame and transpose it
            prediction = model.predict(X_pred)
            
            # Optionally, you can store the prediction in the session
            session['prediction'] = prediction.tolist()

            print(session['prediction'])
            print(prediction)

            # Render the HTML template with variables, prediction, and model parameters
            return render_template('CsvImportAllDatasetNormalizedCorrelated.html', variables=variables, prediction=prediction, parameters=parameters, selected_model=selected_model)

    return render_template("CsvImportAllDatasetNormalizedCorrelated.html", parameters=parameters)
   
@app.route('/AllDataset/AllDatasetPCA', methods=['GET', 'POST'])
def upload_and_show_data_pca():
    parameters = None  # Default value for parameters
    if request.method == 'POST':
        # Handle file upload
        f = request.files.get('file')
        if f and allowed_file(f.filename):
            data_filename = secure_filename(f.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
            f.save(file_path)
            session['uploaded_data_file_path'] = file_path
            model = None
            selected_model = request.form.get('model')

            print(selected_model)

            if selected_model == 'svm':
                model = joblib.load('Models/AllDatasetPCA/SVM.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetPCA/SVM_feature_names.csv")
                print("aqui")
            elif selected_model == 'nb':
                model = joblib.load('Models/AllDatasetPCA/Naive Bayes.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetPCA/Naive Bayes_feature_names.csv")
            elif selected_model == 'knn':
                model = joblib.load('Models/AllDatasetPCA/KNN.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetPCA/kNN_feature_names.csv")
            elif selected_model == 'adaboost':
                model = joblib.load('Models/AllDatasetPCA/AdaBoost.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetPCA/AdaBoost_feature_names.csv")
            elif selected_model == 'dt':
                model = joblib.load('Models/AllDatasetPCA/Decision Tree.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetPCA/Decision Tree_feature_names.csv")
            elif selected_model == 'xgboost':
                model = joblib.load('Models/AllDatasetPCA/XGBoost.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetPCA/XGBoost_feature_names.csv")
            elif selected_model == 'rf':
                model = joblib.load('Models/AllDatasetPCA/Random Forest.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetPCA/Random Forest_feature_names.csv")
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Get the feature names from the model
            # Extract the feature names from the DataFrame

            pca = joblib.load('Models/AllDatasetPCA/PCA.joblib')
            print("Oi")

            # Step 2: Transform new data using the loaded PCA object
            new_data_pca = pca.transform(df)
            print("Oi2")

            #selected_features = model.get_booster().feature_names
            
            if hasattr(model, 'get_params'):
                parameters = model.get_params()
                print("parametros")
            else:
                parameters = None  # or any other way to handle this case

            variables = {col: value for col, value in df.items()}

            X_pred = pd.DataFrame(new_data_pca)  # Use the transformed data directly

            prediction = model.predict(X_pred)
            
            # Optionally, you can store the prediction in the session
            session['prediction'] = prediction.tolist()

            print(session['prediction'])
            print(prediction)

            # Render the HTML template with variables, prediction, and model parameters
            return render_template('CsvImportAllDatasetPCA.html', variables=variables, prediction=prediction, parameters=parameters, selected_model=selected_model)

    return render_template("CsvImportAllDatasetPCA.html", parameters=parameters)


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
    app.run(host='0.0.0.0', port='8080')