# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 11:11:34 2024

@author: vinodsingh
"""

import streamlit as st
import helper
import pickle

model = pickle.load(open('RandomForestModel.pkl','rb'))

st.header('Duplicate Question Pairs')

q1 = st.text_input('Enter Question1')
q2 = st.text_input('Enter Question2')

if st.button('Find'):
    query = helper.query_point_creator(q1, q2)
    result = model.predict(query)
    
    if result:
        st.header('Duplicate')
    else:
        st.header('Not Duplicate')
    
