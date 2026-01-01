import streamlit as st
import pandas as pd 
from sklearn.linear_model import LinearRegression 
 
st.title("Simple stock price  Prediction App")

df = pd.read_csv("last year price.csv")

st.subheader("stock price Dataset")
st.dataframe(df)

model = LinearRegression()
model.fit(df[["Year"]], df["Closing price"])

price = st.number_input("Enter year (in no.):")

pred = model.predict([[price]])

st.subheader("Predicted stock price")
st.write(int(pred[0]))