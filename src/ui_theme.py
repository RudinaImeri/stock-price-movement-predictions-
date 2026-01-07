import streamlit as st


def app_style(df):
    def color_change(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return "color: #00c853"
            elif val < 0:
                return "color: #ff5252"
        return ""

    return (
        df.style
        .applymap(color_change, subset=["Change", "Change %"])
        .format({
            "Price": "{:.2f}",
            "Change %": "{:.2f}%",
            "Volume": "{:,}",
            "Confidence": "{:.3f}"
        })
    )


def apply_global_theme():
    st.markdown("""
        <style>
        /* Page background */
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }

        /* Titles */
        h1, h2, h3 {
            color: #fafafa;
            font-weight: 700;
        }

        /* DataFrame */
        .dataframe {
            font-size: 14px;
        }

        /* Tables */
        table {
            width: 100% !important;
        }

        thead tr th {
            background-color: #161b22 !important;
            color: #c9d1d9 !important;
            font-weight: 600;
            border-bottom: 1px solid #30363d !important;
        }

        tbody tr {
            background-color: #0e1117 !important;
        }

        tbody tr:nth-child(even) {
            background-color: #161b22 !important;
        }

        tbody tr:hover {
            background-color: #1f2933 !important;
        }

        td {
            border-bottom: 1px solid #30363d !important;
            padding: 8px !important;
        }

        /* Metric cards */
        [data-testid="metric-container"] {
            background-color: #161b22;
            border: 1px solid #30363d;
            padding: 12px;
            border-radius: 8px;
        }

        </style>
    """, unsafe_allow_html=True)
