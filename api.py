# Dependencies
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
from feature_engineering import num_to_str
from feature_engineering import impute_categorical
from feature_engineering import impute_garage_bsmt
from feature_engineering import impute_mszoning
from feature_engineering import update_objects
from feature_engineering import impute_lotfrontage
from feature_engineering import update_numerics
from feature_engineering import normalize_skewed
from feature_engineering import new_features
from feature_engineering import get_final_df
from feature_engineering import drop_overfit

# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if xgb:
        try:
            json_ = request.json
            print(json_)
            query = pd.DataFrame(json_)
            query = query.reindex(columns=original_predictors)
            query = num_to_str(query)
            query = impute_categorical(query)
            query = impute_garage_bsmt(query)
            query = impute_mszoning(query, MSZoning_modes)
            query = update_objects(query, objects)
            query = impute_lotfrontage(query, LotFrontage_modes)
            query = update_numerics(query, numerics)
            query = normalize_skewed(query, skew_index, lmbdas)
            query = new_features(query)
            query = get_final_df(query)
            #query = drop_overfit(query, overfit)
            query = query.reindex(columns=model_columns, fill_value=0)
            prediction = list(np.expm1(xgb.predict(query)))
            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    original_predictors = joblib.load('original_predictors.pkl')
    print("Original predictors loaded!")

    MSZoning_modes = joblib.load('MSZoning_modes.pkl') #Load 'MSZoning_modes.pkl'
    print ('MSZoning modes loaded!')

    objects = joblib.load('objects.pkl') #Load 'obkects.pkl'
    print ('Object type columns loaded!')

    LotFrontage_modes = joblib.load('LotFrontage_modes.pkl')
    print("LotFrontage modes loaded!")

    numerics = joblib.load('numerics.pkl')
    print("Numerics loaded!")

    skew_index = joblib.load('skew_index.pkl')
    print("Skew Index loaded!")

    lmbdas = joblib.load('lmbdas.pkl')
    print("Lmbdas loaded!")

    overfit = joblib.load('overfit.pkl')
    print("Overfit loaded!")

    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    xgb = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')

    app.run(port=port, debug=True)
