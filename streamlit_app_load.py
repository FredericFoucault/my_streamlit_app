import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
import plotly.express as px
from zipfile import ZipFile
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
plt.style.use('fivethirtyeight')
#sns.set_style('darkgrid')
import re

# this part was added to connect to the flask-API.py
import requests
import json

@st.cache
def main() :
    data = requests.get('https://flaskapiocr-ec7ba47103cd.herokuapp.com/').text
    data = pd.DataFrame(json.loads(data))
    return data

df = main()
st.write(df)
if __name__ == '__main__':
    main()