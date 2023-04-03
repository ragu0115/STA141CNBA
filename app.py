#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from Model import predict_net_rating

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # get the input values from the form
    age = request.form.get('age')
    player_height = request.form.get('player_height')
    player_weight = request.form.get('player_weight')
    pts = request.form.get('pts')
    reb = request.form.get('reb')
    ast = request.form.get('ast')
    oreb_pct = request.form.get('oreb_pct')
    dreb_pct = request.form.get('dreb_pct')
    usg_pct = request.form.get('usg_pct')
    ts_pct = request.form.get('ts_pct')
    ast_pct = request.form.get('ast_pct')

    # call your prediction function to get the result
    result = predict_net_rating(age, player_height, player_weight, pts, reb, ast, oreb_pct, dreb_pct, usg_pct, ts_pct, ast_pct)

    # render the HTML template with the result
    return render_template('index.html', result=result)

