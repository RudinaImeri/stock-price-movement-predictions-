import streamlit as st
from src.data_preprocessing import load_data, clean_data

st.title("Data Overview")

train, test = load_data("data/train.csv", "data/test.csv")
train = clean_data(train)

st.subheader("Train dataset")
st.dataframe(train.head(100))

st.write("Shape:", train.shape)
st.write("Target distribution:")
st.bar_chart(train["result"].value_counts())
