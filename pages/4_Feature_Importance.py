import streamlit as st
import pandas as pd

st.title("Feature Importance")

if "model" not in st.session_state:
    st.warning("Please train the model first.")
    st.stop()

model = st.session_state["model"]

feature_names = model.feature_names_in_

importances = model.feature_importances_

fi = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.subheader("Top important features")
st.dataframe(fi.head(20))

st.subheader("Importance chart")
st.bar_chart(fi.set_index("Feature").head(20))
