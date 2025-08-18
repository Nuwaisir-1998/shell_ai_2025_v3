import streamlit as st
import pandas as pd
import json
import os
import numpy as np



def load_cv_runs(base_dirs, target_col):
    runs = []
    for base_dir in base_dirs:
        for run_dir in os.listdir(base_dir):
            run_path = os.path.join(base_dir, run_dir)
            if not os.path.isdir(run_path):
                continue

            score_file = os.path.join(run_path, "score.csv")
            params_file = os.path.join(run_path, "params.json")

            if not os.path.exists(score_file) or not os.path.exists(params_file):
                continue

            # read score
            score = float(pd.read_csv(score_file)['Score'].values[0])

            # read params
            with open(params_file, "r") as f:
                params = json.load(f)
            
            seed = ''
            
            if 'seed' in params:
                seed = params['seed']
            
            method_name = base_dir.split('/')[-1]
            
            if params['target_col_name'] == target_col:
                runs.append({
                    "run": f'{run_dir}_seed:{seed} ({method_name})',
                    "score": score,
                    "params": params,
                    "oof_preds": np.load(os.path.join(run_path, 'df_oof_preds.npy')),
                })
    return runs

'View the CV scores and the parameters used for each run:'

with st.expander('TabM'):
    # Load runs
    for target_col in st.session_state['selected_target_cols']:
        BASE_DIR = "./runs/tabm_cv"
        runs = load_cv_runs(base_dirs=[BASE_DIR], target_col=target_col)

        if len(runs) > 0:
            # sort by score descending
            with st.expander(target_col):
                df = pd.DataFrame(runs).sort_values(by="score", ascending=False)

                st.markdown("## Runs Leaderboard")
                st.dataframe(df[["run", "score"]].astype('str'), hide_index=True)

                # Optionally view params of a run
                selected_run = st.selectbox("Select a run to see details:", df["run"])
                params = df[df["run"] == selected_run]["params"].iloc[0]
                st.json(params)
        # else:
            # st.info("No CV runs found yet.")
with st.expander('Autogluon'):
    # Load runs
    for target_col in st.session_state['selected_target_cols']:
        BASE_DIRS = ["./runs/autogluon_best", "./runs/autogluon_exp"]
        runs = load_cv_runs(base_dirs=BASE_DIRS, target_col=target_col)

        if len(runs) > 0:
            # sort by score descending
            with st.expander(target_col):
                df = pd.DataFrame(runs).sort_values(by="score", ascending=False)

                st.markdown("## CV Runs Leaderboard")
                st.dataframe(df[["run", "score"]].astype('str'), hide_index=True)

                # Optionally view params of a run
                selected_run = st.selectbox("Select a run to see details:", df["run"], key=f'{target_col}_select_run')
                params = df[df["run"] == selected_run]["params"].iloc[0]
                st.json(params)
                

with st.expander('Ensembles'):
    # Load runs
    for target_col in st.session_state['selected_target_cols']:
        BASE_DIR = "./runs/ensembles"
        runs = load_cv_runs(base_dirs=[BASE_DIR], target_col=target_col)

        if len(runs) > 0:
            # sort by score descending
            with st.expander(target_col):
                df = pd.DataFrame(runs).sort_values(by="score", ascending=False)

                st.markdown("## CV Runs Leaderboard")
                st.dataframe(df[["run", "score"]].astype('str'), hide_index=True)

                # Optionally view params of a run
                selected_run = st.selectbox("Select a run to see details:", df["run"])
                params = df[df["run"] == selected_run]["params"].iloc[0]
                st.json(params)

