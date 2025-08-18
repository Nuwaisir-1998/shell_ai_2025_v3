import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from glob import glob
import optuna
from sklearn.metrics import mean_absolute_percentage_error
import shutil

def load_oof_files(folder_path):
    oof_list = []
    
    for path in sorted(glob(os.path.join(folder_path, "*.npy"))):
        arr = np.load(path)
        oof_list.append(arr)
    return np.column_stack(oof_list)

def optimize_weights(oof_stack, y_true, n_trials=100):
    def objective(trial):
        weights = np.array([trial.suggest_float(f"w{i}", 0, 1) for i in range(oof_stack.shape[1])])
        weights /= weights.sum()
        pred = np.dot(oof_stack, weights)
        mape = mean_absolute_percentage_error(y_true, pred)  # SV metric
        return 100 - 90 * mape / 2.72

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    best_weights = np.array([study.best_params[f"w{i}"] for i in range(oof_stack.shape[1])])
    best_weights /= best_weights.sum()
    return best_weights

def ensemble_oof_predictions(train_df, oof_root, target_cols, output_dir="ensemble_oof", n_trials=100):
    os.makedirs(output_dir, exist_ok=True)
    # train_df = pd.read_csv(train_csv)
    results = {}

    for target in target_cols:
        print(f"\n▶ Ensembling {target}")
        y_true = train_df[target].values
        folder = os.path.join(oof_root, target)
        oof_stack = load_oof_files(folder)

        print(f"   - {oof_stack.shape[1]} models")
        weights = optimize_weights(oof_stack, y_true, n_trials)
        print(weights)
        final_pred = np.dot(oof_stack, weights)
        # corr = np.corrcoef(final_pred, y_true)[0, 1]
        mape = mean_absolute_percentage_error(y_true, final_pred)  # SV metric
        score = 100 - 90 * mape / 2.72

        np.save(os.path.join(output_dir, f"{target}_ensemble.npy"), final_pred)
        results[target] = {"Score": score, "weights": weights.tolist()}
        print(f"   ✅ Score: {score:.6f}")

    return results


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
            
            seed = 'NA'
            
            if 'seed' in params:
                seed = params['seed']
            
            method_name = base_dir.split('/')[-1]
            
            if params['target_col_name'] == target_col:
                runs.append({
                    "run": f'{run_dir}_seed:{seed} ({method_name})',
                    "score": score,
                    "params": params,
                    "oof_preds": np.load(os.path.join(run_path, 'df_oof_preds.npy')).astype('float64'),
                })
    return runs


BASE_DIRS = ["./runs/tabm_cv", "./runs/autogluon_best", "./runs/ensembles"]

n_top = st.text_input('Number of Top OOF Predictions to use', value=5)
n_top = int(n_top)
st.session_state['n_top'] = n_top

if os.path.exists('./Ensemble_oofs'):
    shutil.rmtree('./Ensemble_oofs')
    
if st.button('Ensemble'):
    for target_col in st.session_state['selected_target_cols']:
        runs = load_cv_runs(base_dirs=BASE_DIRS, target_col=target_col)

        df = pd.DataFrame(runs).sort_values(by="score", ascending=False)
        for i in range(n_top):
            ensemble_oof_path = f"./Ensemble_oofs/{df.iloc[i]['params']['target_col_name']}"
            os.makedirs(ensemble_oof_path, exist_ok=True)
            np.save(f"{ensemble_oof_path}/{df.iloc[i]['run']}_score_{df.iloc[i]['score']}.npy", df.iloc[i]['oof_preds'])
        # st.markdown(f"## {target_col} Runs Leaderboard")
        # st.dataframe(df[["run", "score"]], hide_index=True)
    
    
# target_cols = [f"BlendProperty{i}" for i in range(1, 11)]

    results = ensemble_oof_predictions(
        st.session_state['df_train'],
        oof_root="./Ensemble_oofs",  # or your new directory
        target_cols=st.session_state['selected_target_cols'],
        output_dir="./Ensemble_results",
        n_trials=100
    )

    

    print("\nFinal Scores:")
    for k, v in results.items():
        all_files = glob('./runs/ensembles/ense*')
        latest_file = max([int(folder_name.split('_')[-1]) for folder_name in all_files])
        save_path = f'./runs/ensembles/ensemble_{latest_file + 1}'
        os.makedirs(save_path)
        
        params = {}
        params['target_col_name'] = k
        params['score'] = v['Score']
        params['ensemble_candidates'] = glob(f'./Ensemble_oofs/{k}/*')
        
        
        with open(f"{save_path}/params.json", "w") as f:
            f.write(json.dumps(params, indent=4))
        
        shutil.copy(f'./Ensemble_results/{k}_ensemble.npy', f'{save_path}/df_oof_preds.npy')
        
        pd.DataFrame([v['Score']], columns=['Score']).to_csv(f'{save_path}/score.csv')
        
        print(f"{k}: {v['Score']:.6f}")


# load_cv_runs(BASE_DIRS, target_col)

df_all = pd.DataFrame()

for target_col in st.session_state['selected_target_cols']:
        runs = load_cv_runs(base_dirs=BASE_DIRS, target_col=target_col)
        # target_col, runs
        if len(runs) > 0:
            # sort by score descending
            # with st.expander(target_col):
            df = pd.DataFrame(runs).sort_values(by="score", ascending=False)

            st.markdown(f"## {target_col} CV Runs Leaderboard")
            st.dataframe(df[["run", "score"]].astype('str'), hide_index=True)
            
            selected_run = st.selectbox("Select a run to see details:", df["run"])
            params = df[df["run"] == selected_run]["params"].iloc[0]
            st.json(params)
            
            # col_num = target_col.split('ty')[-1]
            # df_all[f'BP{col_num}'] = df[['run']].iloc[:n_top]
            
# df_all
                # st.markdown("## CV Runs Leaderboard")
                # st.dataframe(df[["run", "score"]], hide_index=True)