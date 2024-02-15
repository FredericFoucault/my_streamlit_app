from flask import Flask, render_template, jsonify,request
import pandas as pd
import pickle
import json
<<<<<<< HEAD

=======
>>>>>>> b965b543590ded6c84e5ad720dbb744cf69144f9
import joblib

app = Flask(__name__)
 
#load the model
#model = pickle.load(open('model_LGBM.pkl','rb'))
<<<<<<< HEAD
model = joblib.load('model_scoring_LGBM.joblib') # lightgbm model
=======
model = joblib.load('model_scoring_HGBC.joblib')
>>>>>>> b965b543590ded6c84e5ad720dbb744cf69144f9

#load the data
df = pd.read_csv("app_clean_final.csv")
dictionnaire = {}
for colonne in df.columns:
    dictionnaire[colonne] = df[colonne].tolist()
 
<<<<<<< HEAD
@app.route('/data',methods =["GET"])
=======
@app.route('/data')
>>>>>>> b965b543590ded6c84e5ad720dbb744cf69144f9
def index():
    return jsonify(dictionnaire)
 

data = pd.DataFrame(dictionnaire)



X=data.drop(['SK_ID_CURR','TARGET'],axis=1)



#@app.route('/prediction', methods=['GET','POST'])
@app.route('/prediction')
def prediction():
    ClientID = request.args.get('ClientID')
    #ClientID = request.form.to_dict('ClientID')
<<<<<<< HEAD
    score = model.predict_proba(X[X.index == int(float(ClientID))])[:,1].tolist()
=======
    score = model.predict_proba(X[X.index == int(ClientID)])[:,1].tolist()
>>>>>>> b965b543590ded6c84e5ad720dbb744cf69144f9
    #score=score[0]*100
    
    score=float(score[0]*100)
    dct= {'ClientID': int(ClientID),
            'prediction':score}
    return jsonify(dct)



<<<<<<< HEAD
@app.route('/', methods=['GET'])
=======
@app.route('/', methods=['GET','POST'])
>>>>>>> b965b543590ded6c84e5ad720dbb744cf69144f9
def menu():
    return render_template('menu.html')

if __name__ == "__main__":


<<<<<<< HEAD
    #modelfile= '../model_LGBM.pkl'
    #model = pickle.load(open(modelfile, 'rb'))
    #print("loaded ok")
    app.run(host="0.0.0.0",port=8080)
=======


    #modelfile= '../model_LGBM.pkl'
    #model = pickle.load(open(modelfile, 'rb'))
    #print("loaded ok")
    app.run(debug=True)
>>>>>>> b965b543590ded6c84e5ad720dbb744cf69144f9
