import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

st.title("Model Evaluation")

if "model" not in st.session_state:
    st.warning("Train the model first.")
    st.stop()

model = st.session_state["model"]

X = st.session_state.get("X_val")
y = st.session_state.get("y_val")

if X is None:
    st.warning("No validation data found.")
    st.stop()

preds = model.predict(X)

cm = confusion_matrix(y, preds)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)
