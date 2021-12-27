from flask import Flask, render_template, request
#import matplotlib.pyplot as plt
#import matplotlib
import joblib
import pandas as pd
import numpy as np
import shap
 

# create Flask App
app = Flask(__name__)  # two underscores here

model = joblib.load("src/churn_model.pkl")

# connect POST API call --> predict() function

@app.route("/", methods=["GET","POST"])       
#"app" here is the Flask app created above
#we will create a local server at http://localhost:5000
#.route('/churn') means when app meets http://localhost:5000/churn
# this whole line means when app meets http://localhost:5000/churn, you should be able to post some data to the api
# the data posted is in JSON format, which looks like a python dictionary 

def predict():
    
    results = ""
    
    if request.method == "POST":
        features = {}
        features["Age"] = float(request.form["Age"])
        features['Dependent#'] = float(request.form['Dependent#'])
        features['MonthsOnBook'] = float(request.form['MonthsOnBook'])
        features['TotalRelationship#'] = float(request.form['TotalRelationship#'])
        features['InactiveMonths'] = float(request.form['InactiveMonths'])
        features['Contact#'] = float(request.form['Contact#'])
        features['CreditLimit'] = float(request.form['CreditLimit'])
        features['RevolvBal'] = float(request.form['RevolvBal'])
        features['TransAmtQ4/Q1'] = float(request.form['TransAmtQ4/Q1'])
        features['TotalTransAmt'] = float(request.form['TotalTransAmt'])
        features['TotalTransCt'] = float(request.form['TotalTransCt'])
        features['TransCtQ4/Q1'] = float(request.form['TransCtQ4/Q1'])
        
        features['RevolvBal/CreditLimit'] = np.round(features['RevolvBal']/features['CreditLimit'], 5)
        features['TotalTransAmt/CreditLimit/12'] = np.round(features['TotalTransAmt']/features['CreditLimit']/12, 5)
        features['AverageCost'] = np.round(features['TotalTransAmt']/features['TotalTransCt'], 3)
  
        df = pd.DataFrame(features, index=[0])
        churn_class = model.predict(df)
        churn_probability = np.round(model.predict_proba(df)[0][1], 3)
               
        
        
        # get shap image. the third line can't run in API, but it works well in Jupyter Notebook
        shap_explainer = shap.TreeExplainer(model)
        shap_v_customer = shap_explainer.shap_values(df.iloc[0])
        p=shap.force_plot(shap_explainer.expected_value[1],shap_v_customer[1],df.iloc[0])
        #plt.savefig('static/image.png', bbox_inches='tight', dpi=100)
        
        results = {'Churn prediction (1-Yes, 0-No)':churn_class[0], 'Churn probability':churn_probability}
        
    return render_template("index.html", result=results)


        
        
# Load model and col_name
if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)