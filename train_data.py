from pygwalker.api.streamlit import StreamlitRenderer
import pandas as pd
import streamlit as st

# Adjust the width of the Streamlit page
st.set_page_config(
    page_title="Use Pygwalker In Streamlit",
    layout="wide"
)

df = st.session_state['df_train']

uploaded_file = st.file_uploader("Upload your train.csv (competition data uploaded by default)")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['df_train'] = df

'train dataframe dimension', df.shape

pyg_app = StreamlitRenderer(df)

pyg_app.explorer()