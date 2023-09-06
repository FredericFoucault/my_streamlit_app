

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.ensemble import (GradientBoostingClassifier, 
                              HistGradientBoostingClassifier)
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import precision_score, recall_score

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import PrecisionRecallDisplay

from sklearn.metrics import RocCurveDisplay

import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.filterwarnings('ignore')

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Application de Machine Learning pour établir un score de credit")
    st.subheader("Auteur : Frederic Foucault")
    
    # Fonction pour importer les données

    def load_data():
        data = pd.read_csv('application_github.csv')
        data.replace(' ', 0, inplace=True)
        return data

    df= load_data()
    
    import re
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    #df.replace('None', 0, inplace=True)
    #df = df.where(pd.notna(df), np.nan)
    df = df.dropna(subset=['TARGET'])
    
    df_sample= df.sample(100)
    df_classifier= df.sample(10000)


    # selection de ligne da la dataframe
    
    def dataframe_with_selections(df):
        df_with_selections = df.copy()
        df_with_selections.insert(0, "Select", False)

        # Get dataframe row-selections from user with st.data_editor
        edited_df = st.data_editor(
            df_with_selections,
            hide_index=True,
            column_config={"Select": st.column_config.CheckboxColumn(required=True)},
            disabled=df.columns,
        )
    
        # Filter the dataframe using the temporary column, then drop the column
        selected_rows = edited_df[edited_df.Select]
        return selected_rows.drop('Select', axis=1)
    

    selection = dataframe_with_selections(df)
    st.write("Votre selection:")
    st.write("Bon client: TARGET =0, Mauvais client TARGET=1")
    st.write(selection)
    
    selection=selection.drop('TARGET',axis=1)
    
    
    
    # affichage
    #if st.sidebar.checkbox('Afficher les Données Brutes', False):
    #    st.subheader('Echantillons de 100 observations')
    #    st.write(df_sample)
    
    st.sidebar.subheader(" Choisissez l'algorithme :")
    
    seed = 926
    # split train and test
    #@st.cache_data(persist=True)
    def split(df):
        y= df['TARGET']
        X= df.drop('TARGET',axis=1)
        X_train,X_test,y_train,y_test = train_test_split(
        X,y,
        test_size= 0.2,
        stratify=y,
        random_state=seed
        )
    X= df_classifier.drop('TARGET',axis=1)
    y= df_classifier['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=926)
    
    class_names=['Bon client','Mauvais client']
    classifier = st.sidebar.selectbox(
    'Algorithmes',
    ('HistGradientBoostingClassifier', 'LightGBM')
    )
    
    # analyse de la performance
    
    def plot_perf(graphes):
        if "Confusion Matrix" in graphes:
            st.subheader('Matrice de Confusion')
            ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            display_labels=class_names
            )
            st.pyplot()
        if "ROC Curve" in graphes:
            st.subheader('Courbe ROC')
            RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test
            )
            st.pyplot()
        if "Precision-Recall Curve" in graphes:
            st.subheader('Courbe Precision Rappel')
            PrecisionRecallDisplay.from_estimator(
            model,
            X_test,
            y_test
            )
            st.pyplot()

    if classifier == "HistGradientBoostingClassifier":
        st.sidebar.subheader('Hyperparamètres du modèle')
        learning_rate= st.sidebar.slider(
            "Choisir le taux d'apprentissage",
            float(0.01),float(1),step=float(0.1)
        )
        max_depth=st.sidebar.slider(
            "Choisir la profondeur",
            1,10,step=1
        )
        min_samples_leaf=st.sidebar.slider(
            "Choisir le nombre minimal de feuilles",
            30,35,step=1
        )
        max_iter=st.sidebar.slider(
            "Choisir le nombre maximal d'iteration",
            1,300,step=1
        )
        
        graphes_perf= st.sidebar.multiselect(
            "Choisir un graphique de performance du model ML",
            ("Confusion Matrix","ROC Curve","Precision-Recall Curve")
            
            )
        if st.sidebar.button('execution',key="classify"):
            st.subheader("Algorithme HistGradientBoostClassifier")
            st.subheader("Résultats")
            # initialisation du modele
            model = HistGradientBoostingClassifier(
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_iter=max_iter
            )
            model.fit(X_train,y_train)
            # predictions
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            #metriques de performance
            accuracy = model.score(X_test,y_test).round(3)
            precision= precision_score(y_test,y_pred).round(3)
            recall = recall_score(y_test,y_pred).round(3)
            auc_score = roc_auc_score(y_test, y_proba[:, 1]).round(3)
            predictions = model.predict(X_test)
            
            st.write("ACCURACY:", accuracy)
            st.write("PRECISION:", precision)
            st.write("RECALL:", recall)
            st.write("ROC_AUC_SCORE:",auc_score)
            # Affichier les graphiques de performance
            plot_perf(graphes_perf)
            
            #faire une prediction
            if len(selection) !=0:
                
                prediction =  model.predict(selection)
                
                st.subheader("Le client est :")
                st.write("Bon client (prediction = 0), Mauvais client (prediction = 1)")
                st.write('Prediction:',prediction)
                

    elif classifier == "LightGBM":

        st.sidebar.subheader('Hyperparamètres du modèle')
        
        feature_fraction=st.sidebar.slider(
            "Choisir la fraction de features",
            float(0.01),float(1),step=float(0.1)
        )
        learning_rate= st.sidebar.slider(
            "Choisir le taux d'apprentissage",
            float(0.01),float(1),step=float(0.1)
        )
        max_depth=st.sidebar.slider(
            "Choisir la profondeur",
            1,10,step=1
        )
        num_leaves=st.sidebar.slider(
            "Choisir le nombre minimal de feuilles",
            int(2),int(100),step=int(1)
        )
        min_data_in_leaf=st.sidebar.slider(
            "Choisir le nombre minimal de donnes par feuille",
            int(1),int(100),step=int(1)
        )
        num_iterations=st.sidebar.slider(
            "Choisir le nombre de boost",
            1,1000,step=10
        )
        subsample=st.sidebar.slider(
            "Choisir le pourcentage de subsample",
            float(0),float(1),step=float(0.01)
        )
        bagging_fraction=st.sidebar.slider(
            "Choisir le bagging fraction",
            float(0.01),float(1),step=float(0.1)
        )
        graphes_perf= st.sidebar.multiselect(
            "Choisir un graphique de performance du model ML",
            ("Confusion Matrix","ROC Curve","Precision-Recall Curve")
            )
            
        if st.sidebar.button('execution',key="classify"):
            st.subheader("Algorithme LightGBM")
            st.subheader("Resultats")
            # initialisation du modele
            model = LGBMClassifier(
            nthread=4,
            feature_fraction=feature_fraction,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            num_iterations=num_iterations,
            subsample=subsample,
            bagging_fraction=bagging_fraction

            )
            model.fit(X_train,y_train, eval_set= [(X_train, y_train), (X_test, y_test)], eval_metric= 'auc', verbose= 200, early_stopping_rounds= 50)
            # predictions
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            #metriques de performance
            accuracy = model.score(X_test,y_test).round(3)
            precision= precision_score(y_test,y_pred).round(3)
            recall = recall_score(y_test,y_pred).round(3)
            auc_score = roc_auc_score(y_test, y_proba[:, 1]).round(3)
            predictions = model.predict(X_test)
            

            st.write("ACCURACY:", accuracy)
            st.write("PRECISION:", precision)
            st.write("RECALL:", recall)
            st.write("ROC_AUC_SCORE:",auc_score)
            # Affichier les graphiques de performance
            plot_perf(graphes_perf)
            
            # faire une prediction sur une ligne
            if len(selection) !=0:
                
                prediction =  model.predict(selection)
                
                st.subheader("Le client est :")
                st.write("Bon client (prediction = 0), Mauvais client (prediction = 1)")
                st.write('Prediction:',prediction)
            

if __name__ == '__main__':
    main()



