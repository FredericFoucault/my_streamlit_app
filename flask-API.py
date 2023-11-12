#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 19:12:18 2023
 
@author: narcy
"""
 
from flask import Flask, render_template, jsonify
import pandas as pd
 
app = Flask(__name__)
 
 
df = pd.read_csv("app_clean_final.csv")
dictionnaire = {}
for colonne in df.columns:
    dictionnaire[colonne] = df[colonne].tolist()
 
@app.route('/')
def index():
    return jsonify(dictionnaire)
 
 
if __name__ == "__main__":
    app.run(debug=True)

