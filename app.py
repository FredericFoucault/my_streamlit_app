from flask import Flask, render_template, jsonify,request
import pandas as pd
import pickle
import json


app = Flask(__name__)
 


#load the data
df = pd.read_csv("app_clean_final.csv")
dictionnaire = {}
for colonne in df.columns:
    dictionnaire[colonne] = df[colonne].tolist()
 
@app.route('/data')
def index():
    return jsonify(dictionnaire)
 

data = pd.DataFrame(dictionnaire)



X=data.drop(['SK_ID_CURR','TARGET'],axis=1)



#@app.route('/prediction', methods=['GET','POST'])
@app.route('/prediction')
def prediction():
    ClientID = request.args.get('ClientID')
    #ClientID = request.form.to_dict('ClientID')
    score = model.predict_proba(X[X.index == int(ClientID)])[:,1].tolist()
    #score=score[0]*100
    
    score=float(score[0]*100)
    dct= {'ClientID': int(ClientID),
            'prediction':score}
    return jsonify(dct)



@app.route('/', methods=['GET','POST'])
def menu():
    return render_template('menu.html')

if __name__ == "__main__":

    #load the model
    model = pickle.load(open('model_LGBM.pkl','rb'))


    #modelfile= '../model_LGBM.pkl'
    #model = pickle.load(open(modelfile, 'rb'))
    #print("loaded ok")
    app.run(debug=True)
