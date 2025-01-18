# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import numpy as np
import plotly.express as px
mport matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.title("Housing App :house:")
st.text("Find your dream home over here")


st.image("house.jpg")

 

df=pd.read_csv('kc_house_data.csv')