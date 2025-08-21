import streamlit as st
import pandas as pd
import numpy as np
import threading
import time
import subprocess
import sys
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm
import os
from glob import glob
import json

from my_tabm import apply_tabm_cv, apply_tabm_cv_tune

st.set_page_config(
    page_title="Shell.ai 25 by Analytic-BD",
    layout="wide"  # This enables wide mode
)


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
                    "method": method_name
                })
    return runs


df_train = pd.read_csv("./dataset/train.csv")
df_test = pd.read_csv("./dataset/test.csv")
df_best = pd.read_csv('./submission_cur_best+tabm_b1234_10fold(cb)_93.92695.csv')

df_test_pred = pd.concat([df_test, df_best], axis=1)

feature_cols = df_train.columns[:55].tolist()
target_cols = df_train.columns[55:].tolist()


if 'df_train' not in st.session_state:
    st.session_state['df_train'] = df_train

if 'df_test' not in st.session_state:
    st.session_state['df_test'] = df_test

if 'feature_cols' not in st.session_state:
    st.session_state['feature_cols'] = feature_cols

# Main page content
st.markdown("# Shell.ai Hackathon 2025")

# st.sidebar.markdown("# Main page")

# col_run_tabm, col2 = st.columns(2)

# with col_run_tabm:
df_train = st.session_state['df_train']
'X_train dimension:', df_train[feature_cols].shape
# 'y_train dimension:', df_train[target_cols].shape

models = ['tabm', 'autogluon']

for model in models:
    if model not in st.session_state:
        st.session_state[model] = {}
        st.session_state[model]['show_sidebar'] = False
        if 'seed_lower' not in st.session_state[model]:
            st.session_state[model]['seed_lower'] = 42
        if 'seed_upper' not in st.session_state[model]:
            st.session_state[model]['seed_upper'] = 42
        if 'n_trials' not in st.session_state[model]:
            st.session_state[model]['n_trials'] = 100
        if 'run_seed_lower' not in st.session_state[model]:
            st.session_state[model]['run_seed_lower'] = 42
        if 'run_seed_upper' not in st.session_state[model]:
            st.session_state[model]['run_seed_upper'] = 42
        if 'n_splits' not in st.session_state[model]:
            st.session_state[model]['n_splits'] = 5
        if 'tuner_splits' not in st.session_state[model]:
            st.session_state[model]['tuner_splits'] = 5


options = target_cols

if 'selected_target_cols' not in st.session_state:
    st.session_state['selected_target_cols'] = target_cols
else:
    selected_target_cols = st.session_state.selected_target_cols
    

selected_target_cols = st.multiselect(
    "Target labels to predict:",
    options,
    default=st.session_state['selected_target_cols'],
)

if 'hparam_ranges' not in st.session_state:
    st.session_state['hparam_ranges'] = {}
        
# selected_targets_checkbox

st.session_state['selected_target_cols'] = selected_target_cols

with st.expander('TabM (Gorishniy et al., ICML (2025)', expanded=True):
    "TabM is a tabular deep learning (DL) MLP-based architecture which relies on efficient ensembling of simultaneously trained MLPs which share most of their weights by default. It demonstrates the best performance among tabular DL models."
    st.markdown("[TabM paper.](https://arxiv.org/pdf/2410.24210)")
    with st.expander('Pipeline'):
        st.image('./pipeline.png')
    # with st.container(height=420):
    cur_model = 'tabm'
    # st.header('TabM')
    # 'CV scores:'
    hparams_all = pd.read_csv("./optuna/tabm_cv/hparams_cv.csv", index_col=0)
    'Best CV Scores (OOF predictions):'
    df_cv_score = pd.DataFrame()
    for target_col in selected_target_cols:
        runs = load_cv_runs(['./runs/tabm_cv', './runs/ensembles'], target_col=target_col)
        
        if len(runs) > 0:
            df = pd.DataFrame(runs).sort_values(by="score", ascending=False)
            df_cv_score['BP'+target_col.split('BlendProperty')[-1]] = [str(df['score'].values[0])]
            # df['params'].iloc[0]
            # df.iloc[0]['params']
            
            df = pd.DataFrame(runs)
            df = df[df['method'] == 'tabm_cv']
            df = df.sort_values(by="score", ascending=False)
            # df.iloc[0]['params']
            best_hparams = df.iloc[0]['params']
            # best_hparams
            
        else:
            hparams = hparams_all[hparams_all['Target'] == target_col]
            df_cv_score['BP'+target_col.split('BlendProperty')[-1]] = str(hparams['Score'])
            best_hparams = hparams.iloc[0].to_dict()
            "This should not be printed."
            
        if f'hparams_{target_col}' not in st.session_state['tabm']:
            st.session_state['tabm'][f'hparams_{target_col}'] = best_hparams
    
    df_cv_score.index = ['CV Score']   
    st.table(df_cv_score)
    
    # if len(selected_target_cols) > 0 :
    #     with st.expander('Hyperparameters'):
    #         hparams_all
        
    # if "show_sidebar_tabm" not in st.session_state:
    #     st.session_state.show_sidebar_tabm = False
        
    
        
    st.markdown('##### Run TabM')


    col_run_seed_lower, col_run_seed_lower_input, col_run_seed_upper, col_run_seed_upper_input, col_n_splits, col_n_splits_input, col_set_hyperparameters, col_run_tabm = st.columns([0.4, 0.4, .4, 0.4, 0.4, 0.4, 1.1, 0.9])
    
    with col_set_hyperparameters:
        if st.button('Set Hyperparameters', use_container_width=True, help='Set hyperparameter values of your choice on the sidebar.'):
            for model in models:
                st.session_state[model]['show_sidebar'] = False
            st.session_state[cur_model]['show_sidebar'] = True
                        
    with col_run_seed_lower:
        'Seed lower:'
    
    with col_run_seed_lower_input:
        run_seed_lower = st.text_input(
                            label='',
                            value=42,  # default = best
                            key=f'tabm_run_seed_lower_input',
                            label_visibility="collapsed",
                        )
        st.session_state['tabm']['run_seed_lower'] = run_seed_lower
    
    with col_run_seed_upper:
        'Seed upper:'
    
    with col_run_seed_upper_input:
        run_seed_upper = st.text_input(
                            label='',
                            value=42,  # default = best
                            key=f'tabm_run_seed_upper_input',
                            label_visibility="collapsed",
                        )
        st.session_state['tabm']['run_seed_upper'] = run_seed_upper
    
    
    with col_n_splits:
        'CV Splits:'
    
    with col_n_splits_input:
        n_splits = st.text_input(
                            label='',
                            value=5,
                            key=f'n_splits_input',
                            label_visibility="collapsed",
                        )
        st.session_state['tabm']['n_splits'] = n_splits
    
    with col_run_tabm:
        if st.button('Run Tabm', use_container_width=True, help='Runs TabM one by one for each of the selected targets starting from seed lower up to seed upper.'):
            for target_col in selected_target_cols:
                run_seed_lower = int(st.session_state['tabm']['run_seed_lower'])
                run_seed_upper = int(st.session_state['tabm']['run_seed_upper'])
                for seed in range(run_seed_lower, run_seed_upper + 1):
                    hparams = st.session_state['tabm'][f'hparams_{target_col}']
                    hparams['k'] = 32
                    # hparams
                    df_train = st.session_state['df_train']
                    feature_cols = st.session_state['feature_cols']
                    
                    # progress_bar = st.sidebar.progress(0)
                    # status_text = st.sidebar.empty()
                    st.session_state.status_text_run_tabm.text("Running...")
                    def update_progress(current, total):
                        percent = current / total
                        st.session_state.progress_bar_run_tabm.progress(percent)
                        st.session_state.status_text_run_tabm.text(f"Processing fold {current}/{total}")
                        
                    n_splits = int(st.session_state['tabm']['n_splits'])
                    
                    score, _, _ = apply_tabm_cv(hparams, df_train, df_test_pred, feature_cols, target_col, seed=seed, n_splits=n_splits, callback=update_progress)
                    score
        
    '##### Tune Hyperparameters'
    with st.expander("Set Hyperparameter Tuning Ranges", expanded=False):

        # n_trials (Optuna)
        n_trials = st.number_input("Number of Optuna trials (n_trials)", min_value=1, value=100, step=10, key="n_trials")

        # n_bins
        c1, c2 = st.columns(2)
        with c1:
            n_bins_min = st.number_input("n_bins (min)", min_value=1, value=2, key="n_bins_min")
        with c2:
            n_bins_max = st.number_input("n_bins (max)", min_value=n_bins_min, value=128, key="n_bins_max")

        # d_embedding
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            d_embedding_min = st.number_input("d_embedding (min)", min_value=1, value=8, key="d_embedding_min")
        with c2:
            d_embedding_max = st.number_input("d_embedding (max)", min_value=d_embedding_min, value=32, key="d_embedding_max")
        with c3:
            d_embedding_step = st.number_input("step", min_value=1, value=4, key="d_embedding_step")

        # n_blocks
        c1, c2 = st.columns(2)
        with c1:
            n_blocks_min = st.number_input("n_blocks (min)", min_value=1, value=1, key="n_blocks_min")
        with c2:
            n_blocks_max = st.number_input("n_blocks (max)", min_value=n_blocks_min, value=4, key="n_blocks_max")

        # d_block
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            d_block_min = st.number_input("d_block (min)", min_value=1, value=64, key="d_block_min")
        with c2:
            d_block_max = st.number_input("d_block (max)", min_value=d_block_min, value=1024, key="d_block_max")
        with c3:
            d_block_step = st.number_input("step", min_value=1, value=16, key="d_block_step")

        # embedding_type (categorical)
        c1, c2 = st.columns(2)
        with c1:
            embedding_type = st.multiselect(
                "embedding_type options", 
                ["PeriodicEmbeddings", "PiecewiseLinearEmbeddings"],
                default=["PeriodicEmbeddings", "PiecewiseLinearEmbeddings"]
            )
        with c2:
            arch_type = st.multiselect(
                "arch_type options", 
                ["tabm", "tabm-mini"],
                default=["tabm", "tabm-mini"]
            )

        # lr
        c1, c2 = st.columns(2)
        with c1:
            lr_min = st.number_input("lr (min)", min_value=1e-6, value=1e-4, format="%.1e", key="lr_min")
        with c2:
            lr_max = st.number_input("lr (max)", min_value=lr_min, value=5e-3, format="%.1e", key="lr_max")
        lr_log = st.checkbox("Log scale for lr", value=True, key="lr_log")

        # weight_decay
        c1, c2 = st.columns(2)
        with c1:
            wd_min = st.number_input("weight_decay (min)", min_value=1e-6, value=1e-4, format="%.1e", key="wd_min")
        with c2:
            wd_max = st.number_input("weight_decay (max)", min_value=wd_min, value=1e-1, format="%.1e", key="wd_max")
        wd_log = st.checkbox("Log scale for weight_decay", value=True, key="wd_log")

        # share_training_batches
        share_training_batches = st.multiselect(
            "share_training_batches options",
            ["T", "F"],
            default=["T", "F"],
            key="share_training_batches"
        )
    
    hparam_ranges = {
        "n_bins": (n_bins_min, n_bins_max),
        "d_embedding": (d_embedding_min, d_embedding_max, d_embedding_step),
        "n_blocks": (n_blocks_min, n_blocks_max),
        "d_block": (d_block_min, d_block_max, d_block_step),
        "embedding_type": embedding_type,
        "arch_type": arch_type,
        "lr": (lr_min, lr_max, {"log": lr_log}),
        "weight_decay": (wd_min, wd_max, {"log": wd_log}),
        "share_training_batches": share_training_batches,
        "n_trials": n_trials,
    }
    
    st.session_state['hparam_ranges'] = hparam_ranges

        

    # st.write("### Current Hyperparameters")
    # st.json({
    #     "n_bins": (n_bins_min, n_bins_max),
    #     "d_embedding": (d_embedding_min, d_embedding_max, d_embedding_step),
    #     "n_blocks": (n_blocks_min, n_blocks_max),
    #     "d_block": (d_block_min, d_block_max, d_block_step),
    #     "embedding_type": embedding_type,
    #     "arch_type": arch_type,
    #     "lr": (lr_min, lr_max, {"log": lr_log}),
    #     "weight_decay": (wd_min, wd_max, {"log": wd_log}),
    #     "share_training_batches": share_training_batches,
    #     "n_trials": n_trials,
    # })
    
    col_seed_lower, col_seed_lower_input, col_seed_upper, col_seed_upper_input, col_tuner_splits, col_tuner_splits_input, col_tune_hyperparameters = st.columns([1, 1, 1, 1, 1, 1, 5])
    
    with col_tune_hyperparameters:
        df_best_hparams = None
        if st.button('Start tuning', use_container_width=True, help='Optuna TPESampler is used for sampling hyperparameters on each trial.'):
            
            # n_trials = int(st.session_state['tabm']['n_trials'])
            n_trials = int(hparam_ranges['n_trials'])
            tuner_splits = int(st.session_state['tabm']['tuner_splits'])
            seed_lower = int(st.session_state['tabm']['seed_lower'])
            seed_upper = int(st.session_state['tabm']['seed_upper'])
            
            for seed in range(seed_lower, seed_upper + 1):
                for target_col in selected_target_cols:
                    def objective(trial):
                        score = apply_tabm_cv_tune(trial, df_train, df_test_pred, feature_cols, target_col, seed=100, n_splits=tuner_splits, hparam_ranges=hparam_ranges)
                        return score
                    
                    progress_bar = st.sidebar.progress(0)
                    status_text = st.sidebar.empty()
                    
                    def streamlit_callback(study, trial):
                        completed = len(study.trials)
                        progress_bar.progress(completed / n_trials)
                        status_text.text(f"Running trial {completed}/{n_trials}")

                    study = optuna.create_study(sampler=TPESampler(), direction='maximize')
                    study.optimize(objective, n_trials=n_trials, callbacks=[streamlit_callback])
                    status_text.text("✅ Done!")
                    st.write("Best trial:", study.best_trial.params)
                    
                    map_hparams = study.best_params
                    map_hparams['seed'] = seed
                    map_hparams['Target'] = target_col
                    map_hparams['Score'] = study.best_value
                    map_hparams['Best trial'] = study.best_trial.number
                    df_cur_best = pd.DataFrame([map_hparams])
                    df_best_hparams = pd.concat([df_best_hparams, df_cur_best])
                    os.makedirs('./optuna/tabm_cv', exist_ok=True)
                    
                    hparam_files = glob('./optuna/tabm_cv/*_v*')
                    
                    latest_run = max([int(file_name.split('_v')[-1].split('.')[0]) for file_name in hparam_files])
                    
                    df_best_hparams.to_csv(f'./optuna/tabm_cv/hparams_cv_v{latest_run + 1}.csv')
        
    with col_seed_lower:
        'Seed lower:'

    with col_seed_lower_input:
        
        seed_lower = st.text_input(
                            label='',
                            value=42,  # default = best
                            key=f'tabm_seed_lower',
                            label_visibility="collapsed",
                        )
        st.session_state['tabm']['seed_lower'] = seed_lower
    
    with col_seed_upper:
        'Seed upper:'

    with col_seed_upper_input:
        seed_upper = st.text_input(
                            label='',
                            value=42,  # default = best
                            key=f'tabm_seed_upper',
                            label_visibility="collapsed",
                        )
        st.session_state['tabm']['seed_upper'] = seed_upper
    
    # with col_set_hparam_ranges:
        
    # with col_n_trials:
    #     '\# of trials:'
    
    # with col_n_trials_input:
    #     n_trials = st.text_input(
    #                         label='',
    #                         value=100,  # default = best
    #                         key=f'tabm_n_trials',
    #                         label_visibility="collapsed",
    #                     )
    #     st.session_state['tabm']['n_trials'] = n_trials
    
    with col_tuner_splits:
        'CV Splits:' 
    
    with col_tuner_splits_input:
        tuner_splits = st.text_input(
                            label='',
                            value=5,  # default = best
                            key=f'tabm_tuner_splits',
                            label_visibility="collapsed",
                        )
        st.session_state['tabm']['tuner_splits'] = tuner_splits
   
    

    
        
    with st.sidebar:
        if st.session_state['tabm']['show_sidebar']:
            st.markdown("## Choose TabM Hyperparameters")
            for target_col in selected_target_cols:
                with st.expander(target_col):
                    best_hparams = hparams_all[hparams_all['Target'] == target_col].iloc[0].to_dict()
                    best_hparams.pop('Score')
                    best_hparams.pop('Target')
                    # st.write(best_hparams)
                    
                    # for key in best_hparams.keys():
                        # st.write(key)
                        
                        
                    # Possible categorical options
                    categorical_options = {
                        "embedding_type": ["PeriodicEmbeddings", "PiecewiseLinearEmbeddings"],
                        "arch_type": ['tabm', 'tabm-mini'],
                        "share_training_batches": ['T', 'F'],
                    }

                    # Detect categorical vs numeric
                    categorical_keys = list(categorical_options.keys())
                    numeric_keys = [k for k in best_hparams.keys() if k not in categorical_keys]

                    # st.sidebar.markdown("### Hyperparameter Tuning")

                    # Store updated params
                    updated_hparams = {}

                    # Categorical: dropdown
                    for key in categorical_keys:
                        updated_hparams[key] = st.selectbox(
                            label=key,
                            options=categorical_options[key],
                            index=categorical_options[key].index(best_hparams[key]),  # default = best
                            key=f'{target_col}_{key}_selectbox',
                        )

                    # # Numerical: text input (could use number_input too)
                    for key in numeric_keys:
                        updated_hparams[key] = st.text_input(
                            label=key,
                            value=str(best_hparams[key]),  # default = best
                            key=f'{target_col}_{key}_text_input',
                        )

                    # st.write("Updated Hyperparameters:", updated_hparams)
                    
                    st.session_state['tabm'][f'hparams_{target_col}'] = updated_hparams
        
        # if st.session_state['autogluon']['show_sidebar']:
        #     'Autogluon'


progress_bar_run_tabm = st.empty()
status_text_run_tabm = st.empty()
if "progress" not in st.session_state:
    st.session_state.progress_bar_run_tabm = progress_bar_run_tabm.progress(0)
    
if "status_text_run_tabm" not in st.session_state:
    st.session_state.status_text_run_tabm = status_text_run_tabm.text('')
# if 'status_text_run_tabm' not in st.session_state:
#     st.session_state.status_text_run_tabm = status_text_run_tabm.
# status_text_run_tabm = st.empty()
    
# with st.expander('Autogluon'):
#     cur_model = 'autogluon'
#     ag_preset = st.selectbox("Preset", ['best quality', 'experimental quality'])
#     ag_time = st.text_input("Max time per target (sec)", value=600)
#     if st.button('Run'):
#         # from autogluon.tabular import TabularPredictor
#         from tqdm import tqdm



# st.write("### Current Hyperparameters")
# st.json({
#     "n_bins": (n_bins_min, n_bins_max),
#     "d_embedding": (d_embedding_min, d_embedding_max, d_embedding_step),
#     "n_blocks": (n_blocks_min, n_blocks_max),
#     "d_block": (d_block_min, d_block_max, d_block_step),
#     "embedding_type": embedding_type,
#     "arch_type": arch_type,
#     "lr": (lr_min, lr_max, {"log": lr_log}),
#     "weight_decay": (wd_min, wd_max, {"log": wd_log}),
#     "share_training_batches": share_training_batches,
#     "n_trials": n_trials,
# })
    
                    
            
        
    
   
            


