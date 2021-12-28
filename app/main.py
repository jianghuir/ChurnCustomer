from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import shap
from PIL import Image
 

# create Flask App
app = Flask(__name__)  # two underscores here

model = joblib.load("src/churn_model.pkl")


#we will create a local server at http://localhost:5000
@app.route("/", methods=["GET","POST"])       

def predict():
    
    pred = ""
    churn_probability = ""
    shap_html = Image.open("static/Blank.jpeg")
    note1 = ""
    note2 = ""
    note3 = ""
    
    if request.method == "POST":
        # get customer info from index.html
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
        
        # create three new features
        features['RevolvBal/CreditLimit'] = np.round(features['RevolvBal']/features['CreditLimit'], 5)
        features['TotalTransAmt/CreditLimit/12'] = np.round(features['TotalTransAmt']/features['CreditLimit']/12, 5)
        features['AverageCost'] = np.round(features['TotalTransAmt']/features['TotalTransCt'], 3)
  
        # use model to predict
        df = pd.DataFrame(features, index=[0])
        churn_class = model.predict(df)
        churn_probability = np.round(model.predict_proba(df)[0][1], 3)
               
        if churn_class[0] == 1:
            pred = "Yes"
        else:
            pred = "No"
        
        # get shap image and print on the webpage
        shap_explainer = shap.TreeExplainer(model)
        shap_v_customer = shap_explainer.shap_values(df.iloc[0])
        p=shap.force_plot(shap_explainer.expected_value[1],shap_v_customer[1],df.iloc[0])
        shap_html = f"<head>{shap.getjs()}</head><body>{p.html()}</body>"
        
        note1 = "f(x) value (in bold font) is the predicted churning probability of the customer, ranging from 0.0-1.0. For example, with f(x) 0.90, the customer is predicted to have a churning probability of 90%"
        note2 = "Red blocks are factors leading to a higher churning probability. Blue blocks are factors leading to a lower churning probability."
        note3 = "A bigger block poses a higher influence on churning probability"

    return render_template("index.html", result1=pred, result2=churn_probability, plot_cust=shap_html, imageNote1=note1, imageNote2=note2, imageNote3=note3)


        
        
# Load model
if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)