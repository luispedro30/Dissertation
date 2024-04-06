from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for, session, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import pickle
import os
import joblib
from os.path import join
from werkzeug.utils import secure_filename
import pyaudio
import wave
import featuresscript
import predictscript


app = Flask(__name__)

stream = None
frames = []


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

@app.route('/audio', methods=['GET', 'POST'])
@login_required
def audio():
    return render_template('audio.html')

def record_audio(file_name, duration, sample_rate=44100, chunk_size=1024, format=pyaudio.paInt16, channels=1):
    audio = pyaudio.PyAudio()
    
    stream = audio.open(format=format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)
    
    frames = []
    num_frames = int(sample_rate * duration / chunk_size)

    print("Recording...")
    for i in range(num_frames):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio as a .wav file
    wf = wave.open(file_name, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

@app.route('/start', methods=['GET', 'POST'])
def recordAudio():
    record_audio("a.wav",6)
    return render_template('audio.html')

@app.route('/predict', methods=['GET', 'POST'])
def predictAudio():
    print("Predicting")
    features = featuresscript.compute_features("a.wav")
    prediction =  predictscript.predict(features)
    print(prediction)

    return render_template('audio.html', prediction=prediction)


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

@app.route('/AllDataset/AllDatasetSmoteNormalized', methods=['GET', 'POST'])
def upload_and_show_data_smote_normalized():
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
                model = joblib.load('Models/AllDatasetWithSmoteNormalized/SVM.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetWithSmoteNormalized/SVM_feature_names.csv")
                print("aqui")
            elif selected_model == 'nb':
                model = joblib.load('Models/AllDatasetWithSmoteNormalized/Naive Bayes.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetWithSmoteNormalized/Naive Bayes_feature_names.csv")
            elif selected_model == 'knn':
                model = joblib.load('Models/AllDatasetWithSmoteNormalized/KNN.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetWithSmoteNormalized/kNN_feature_names.csv")
            elif selected_model == 'adaboost':
                model = joblib.load('Models/AllDatasetWithSmoteNormalized/AdaBoost.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetWithSmoteNormalized/AdaBoost_feature_names.csv")
            elif selected_model == 'dt':
                model = joblib.load('Models/AllDatasetWithSmoteNormalized/Decision Tree.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetWithSmoteNormalized/Decision Tree_feature_names.csv")
            elif selected_model == 'xgboost':
                model = joblib.load('Models/AllDatasetWithSmoteNormalized/XGBoost.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetWithSmoteNormalized/XGBoost_feature_names.csv")
            elif selected_model == 'rf':
                model = joblib.load('Models/AllDatasetWithSmoteNormalized/Random Forest.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/AllDatasetWithSmoteNormalized/Random Forest_feature_names.csv")
            
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
            return render_template('CsvImportAllDatasetSmoteNormalized.html', variables=variables, prediction=prediction, parameters=parameters, selected_model=selected_model)

    return render_template("CsvImportAllDatasetSmoteNormalized.html", parameters=parameters)


@app.route('/AllDataset/AllDatasetXGBoost', methods=['GET', 'POST'])
def upload_and_show_data_xgboost():
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
                model = joblib.load('Models/XGBoostFeatureSelection/SVM.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/XGBoostFeatureSelection/SVM_feature_names.csv")
                print("aqui")
            elif selected_model == 'nb':
                model = joblib.load('Models/XGBoostFeatureSelection/Naive Bayes.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/XGBoostFeatureSelection/Naive Bayes_feature_names.csv")
            elif selected_model == 'knn':
                model = joblib.load('Models/XGBoostFeatureSelection/KNN.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/XGBoostFeatureSelection/kNN_feature_names.csv")
            elif selected_model == 'adaboost':
                model = joblib.load('Models/XGBoostFeatureSelection/AdaBoost.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/XGBoostFeatureSelection/AdaBoost_feature_names.csv")
            elif selected_model == 'dt':
                model = joblib.load('Models/XGBoostFeatureSelection/Decision Tree.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/XGBoostFeatureSelection/Decision Tree_feature_names.csv")
            elif selected_model == 'xgboost':
                model = joblib.load('Models/XGBoostFeatureSelection/XGBoost.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/XGBoostFeatureSelection/XGBoost_feature_names.csv")
            elif selected_model == 'rf':
                model = joblib.load('Models/XGBoostFeatureSelection/Random Forest.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/XGBoostFeatureSelection/Random Forest_feature_names.csv")
            
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
            return render_template('CsvImportXGBoostFeatureSelection.html', variables=variables, prediction=prediction, parameters=parameters, selected_model=selected_model)

    return render_template("CsvImportXGBoostFeatureSelection.html", parameters=parameters)



@app.route('/AllDataset/EnsembleStacking', methods=['GET', 'POST'])
def upload_and_show_data_ensemble():
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
                model = joblib.load('Models/XGBoostFeatureSelection/SVM.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/XGBoostFeatureSelection/SVM_feature_names.csv")
                print("aqui")
            elif selected_model == 'nb':
                model = joblib.load('Models/XGBoostFeatureSelection/Naive Bayes.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/XGBoostFeatureSelection/Naive Bayes_feature_names.csv")
            elif selected_model == 'knn':
                model = joblib.load('Models/XGBoostFeatureSelection/KNN.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/XGBoostFeatureSelection/kNN_feature_names.csv")
            elif selected_model == 'adaboost':
                model = joblib.load('Models/XGBoostFeatureSelection/AdaBoost.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/XGBoostFeatureSelection/AdaBoost_feature_names.csv")
            elif selected_model == 'dt':
                model = joblib.load('Models/XGBoostFeatureSelection/Decision Tree.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/XGBoostFeatureSelection/Decision Tree_feature_names.csv")
            elif selected_model == 'xgboost':
                model = joblib.load('Models/XGBoostFeatureSelection/XGBoost.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/XGBoostFeatureSelection/XGBoost_feature_names.csv")
            elif selected_model == 'rf':
                model = joblib.load('Models/XGBoostFeatureSelection/Random Forest.joblib')
                feature_names = pd.read_csv("D:/Dissertation/Models/XGBoostFeatureSelection/Random Forest_feature_names.csv")
            
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
            return render_template('CsvImportXGBoostFeatureSelection.html', variables=variables, prediction=prediction, parameters=parameters, selected_model=selected_model)

    return render_template("CsvImportXGBoostFeatureSelection.html", parameters=parameters)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port='8080')