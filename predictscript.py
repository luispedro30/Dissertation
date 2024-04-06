import pandas as pd
import joblib

def predict(features):

    data = pd.DataFrame.from_dict(features, orient='index', columns=['Value'])


    model = joblib.load('D:/Dissertation/Models/XGBoostFeatureSelection/SVM.joblib')
    feature_names = pd.read_csv("D:/Dissertation/Models/XGBoostFeatureSelection/SVM_feature_names.csv")


    # Assuming features is your dictionary
    data = pd.DataFrame.from_dict(features, orient='index', columns=['Value'])

    # Ensure the index is sorted to avoid potential KeyError
    data = data.sort_index()

    # Now you can proceed with accessing the selected features
    selected_features = feature_names['Feature Names'].tolist()
    selected_features = [feat for feat in selected_features if feat in data.index]

    if selected_features:
        # Access the selected features from the DataFrame
        data_selected = data.loc[selected_features]
        print(data_selected)
    else:
        print("No selected features found in the DataFrame.")
        
    X_pred = pd.DataFrame(data_selected).transpose()  # Convert data to DataFrame and transpose it
    prediction = model.predict(X_pred)
    if prediction == 0:
        return "Não tem doença de Parkinson"
    else:
        return "Tem doença de Parkinson"
