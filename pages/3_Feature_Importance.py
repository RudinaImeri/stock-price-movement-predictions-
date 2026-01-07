import streamlit as st
import pandas as pd
from src.ui_theme import apply_global_theme

apply_global_theme()
st.title("Feature Importance")

model = st.session_state.get("model")
features = st.session_state.get("features")

if model is None:
    st.warning("Train model first")
    st.stop()

fi = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

st.dataframe(fi, use_container_width=True)
st.bar_chart(fi.set_index("Feature"))
