import os
import sys
import threading
import time
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from flask import Flask, render_template_string, jsonify, request, send_from_directory
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             cohen_kappa_score, confusion_matrix, roc_curve, auc)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import logging
import joblib
import shap

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- 群智能优化库 ----------
try:
    from mealpy import FloatVar, IntegerVar
    from mealpy.swarm_based import GWO, SSA, PSO, WOA
    from mealpy.evolutionary_based import GA
    MEALPY_AVAILABLE = True
except ImportError:
    MEALPY_AVAILABLE = False
    logging.warning("⚠️ mealpy 未安装，将跳过群智能优化部分。")

# ---------- TabNet ----------
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    logging.warning("⚠️ pytorch-tabnet 未安装，TabNet 将替换为占位模型。")

# ====================== 配置 ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.config['SECRET_KEY'] = 'sheep_rumen_pro'
app.config['RESULTS_FOLDER'] = os.path.join(BASE_DIR, 'results')
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

_training_status = {}
_status_lock = threading.Lock()

def update_status(task_id: str, **kwargs):
    with _status_lock:
        if task_id not in _training_status:
            _training_status[task_id] = {}
        _training_status[task_id].update(kwargs)

def get_status(task_id: str = 'default'):
    with _status_lock:
        return _training_status.get(task_id, {}).copy()

LABEL_NAMES = ["正常", "轻度萎靡", "应激状态", "重度萎靡"]
FEATURE_COLS = ["temp", "rumen", "step", "env"]

# ====================== 数据生成与加载 ======================
def generate_mock_data():
    n = 200
    np.random.seed(42)
    def gen(mu_temp, mu_rumen, mu_step, mu_env):
        return pd.DataFrame({
            "temp": np.random.normal(mu_temp, 0.3, n),
            "rumen": np.random.normal(mu_rumen, 8, n),
            "step": np.random.normal(mu_step, 20, n),
            "env": np.random.normal(mu_env, 2, n)
        })
    df0 = gen(38.5, 50, 100, 20)
    df1 = gen(39.0, 45, 85, 21)
    df2 = gen(39.5, 35, 130, 22)
    df3 = gen(40.5, 20, 40, 24)

    for df in [df0, df1, df2, df3]:
        df["rumen"] = df["rumen"].clip(lower=0)
        df["step"] = df["step"].clip(lower=0)
        df["temp"] = df["temp"].clip(lower=35, upper=43)

    df0["label"] = 0
    df1["label"] = 1
    df2["label"] = 2
    df3["label"] = 3
    return pd.concat([df0, df1, df2, df3], ignore_index=True)

def load_data():
    try:
        df1 = pd.read_csv("羊-轻度萎靡.txt", header=None)
        df2 = pd.read_csv("羊-应激状态.txt", header=None)
        df3 = pd.read_csv("羊-正常状态.txt", header=None)
        df4 = pd.read_csv("羊-重度萎靡.txt", header=None)
        df1.columns = df2.columns = df3.columns = df4.columns = FEATURE_COLS
        df1["label"] = 1
        df2["label"] = 2
        df3["label"] = 0
        df4["label"] = 3
        df = pd.concat([df1, df2, df3, df4], ignore_index=True)
        for col in FEATURE_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna()
    except Exception as e:
        logging.warning(f"真实数据加载失败，使用模拟数据: {e}")
        return generate_mock_data()

def feature_engineering(df):
    df = df.copy()
    df["temp_diff"] = df["temp"] - df["env"]
    df["activity_ratio"] = df["step"] / (df["rumen"] + 1e-5)
    return df

def preprocess(df, test_size=0.1, val_ratio=0.1111, random_state=42):
    model_features = FEATURE_COLS + ["temp_diff", "activity_ratio"]
    X = df[model_features]
    y = df["label"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state, stratify=y_train_val
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, model_features, scaler

# ====================== 模型包装类 ======================
class ModelWrapper:
    def __init__(self, name, model_obj, is_nn=False, input_dim=None, output_dim=4):
        self.name = name
        self.model = model_obj
        self.is_nn = is_nn
        self.input_dim = input_dim
        self.output_dim = output_dim

    def fit(self, X, y):
        y = np.array(y).ravel()
        if self.is_nn:
            return self._fit_nn(X, y)
        else:
            self.model.fit(X, y)
            return self

    def predict(self, X):
        if self.is_nn:
            return self._predict_nn(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.is_nn:
            return self._predict_proba_nn(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        pred = self.model.predict(X)
        proba = np.zeros((len(pred), self.output_dim))
        proba[np.arange(len(pred)), pred] = 1.0
        return proba

    def _fit_nn(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        for _ in range(30):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def _predict_nn(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            return torch.argmax(self.model(X_t), dim=1).numpy()

    def _predict_proba_nn(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            return torch.softmax(self.model(X_t), dim=1).numpy()

# ====================== 神经网络结构 ======================
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class CNN1D(nn.Module):
    def __init__(self, input_dim, output_dim=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(16 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.conv(x.unsqueeze(1))

def get_all_models(input_dim):
    models = {
        "RandomForest": ModelWrapper("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
        "XGBoost": ModelWrapper("XGBoost", xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')),
        "LightGBM": ModelWrapper("LightGBM", lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)),
        "CatBoost": ModelWrapper("CatBoost", cb.CatBoostClassifier(n_estimators=100, random_state=42, verbose=0)),
        "SVM": ModelWrapper("SVM", SVC(kernel='rbf', probability=True, random_state=42)),
        "KNN": ModelWrapper("KNN", KNeighborsClassifier(n_neighbors=5)),
        "MLP": ModelWrapper("MLP", MLP(input_dim), is_nn=True, input_dim=input_dim),
        "1D-CNN": ModelWrapper("1D-CNN", CNN1D(input_dim), is_nn=True, input_dim=input_dim)
    }
    if TABNET_AVAILABLE:
        models["TabNet"] = ModelWrapper("TabNet", TabNetClassifier(verbose=0))
    else:
        models["TabNet(placeholder)"] = ModelWrapper("TabNet", RandomForestClassifier(n_estimators=50, random_state=42))
    return models

# ====================== 优化算法模块 ======================
def objective_func(params, X, y):
    p = {
        'n_estimators': int(params[0]),
        'max_depth': int(params[1]),
        'learning_rate': params[2],
        'subsample': params[3],
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'n_jobs': -1
    }
    model = xgb.XGBClassifier(**p)
    return cross_val_score(model, X, y, cv=3, scoring='f1_weighted').mean()

def run_6_optimization_battle(X_train, y_train, task_id):
    results = []

    # 1. 贝叶斯优化
    update_status(task_id, message="🔵 正在进行贝叶斯优化...")
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective_func([
            trial.suggest_int('n_estimators', 50, 150),
            trial.suggest_int('max_depth', 3, 7),
            trial.suggest_float('learning_rate', 0.01, 0.2),
            trial.suggest_float('subsample', 0.7, 1.0)
        ], X_train, y_train),
        n_trials=10
    )
    results.append({
        'opt_type': 'Bayesian',
        'cv_f1': study.best_value,
        'params': study.best_params
    })

    # 2. 群智能优化 (mealpy 3.x 适配)
    if MEALPY_AVAILABLE:
        bounds = [
            IntegerVar(lb=50, ub=150, name="n_estimators"),
            IntegerVar(lb=3, ub=7, name="max_depth"),
            FloatVar(lb=0.01, ub=0.2, name="learning_rate"),
            FloatVar(lb=0.7, ub=1.0, name="subsample")
        ]

        def mealpy_obj(sol):
            return -objective_func(sol, X_train, y_train)

        problem = {
            "obj_func": mealpy_obj,
            "bounds": bounds,
            "minmax": "min"
        }

        optimizers = [
            ("GWO", GWO.OriginalGWO(epoch=5, pop_size=10), "灰狼优化 (GWO)"),
            ("SSA", SSA.OriginalSSA(epoch=5, pop_size=10), "麻雀搜索 (SSA)"),
            ("PSO", PSO.OriginalPSO(epoch=5, pop_size=10), "粒子群优化 (PSO)"),
            ("GA",  GA.BaseGA(epoch=5, pop_size=10),      "遗传算法 (GA)"),
            ("WOA", WOA.OriginalWOA(epoch=5, pop_size=10), "鲸鱼优化 (WOA)")
        ]

        for name, opt_inst, chinese_name in optimizers:
            try:
                update_status(task_id, message=f"🟡 正在进行{chinese_name}...")
                agent = opt_inst.solve(problem)
                if agent is None:
                    continue
                sol = agent.solution
                # 关键修复：用 .target.fitness 获取数值
                results.append({
                    'opt_type': name,
                    'cv_f1': -agent.target.fitness,
                    'params': {
                        'n_estimators': int(sol[0]),
                        'max_depth': int(sol[1]),
                        'learning_rate': sol[2],
                        'subsample': sol[3]
                    }
                })
            except Exception as e:
                logging.warning(f"{chinese_name} 优化失败: {e}")

    # 3. 绘制对比图
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(x='opt_type', y='cv_f1', data=df_res, palette='magma')
        plt.title("6种优化算法在 XGBoost 上的 F1 分数对比")
        plt.ylim(0.5, 1.0)
        plt.tight_layout()
        opt_img_path = 'opt_comparison.png'
        plt.savefig(os.path.join(app.config['RESULTS_FOLDER'], opt_img_path), dpi=300)
        plt.close()

        best_idx = df_res['cv_f1'].idxmax()
        best_info = df_res.loc[best_idx]
        return results, opt_img_path, best_info
    else:
        # 回退默认参数
        default_params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 1.0}
        best_info = pd.Series({'opt_type': 'default', 'cv_f1': 0.0, 'params': default_params})
        return [], None, best_info

# ====================== 图表生成函数 ======================
def save_confusion_matrix(y_true, y_pred, model_name, folder):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title(f'{model_name} 混淆矩阵')
    plt.tight_layout()
    filename = f"{model_name}_confusion.png"
    plt.savefig(os.path.join(folder, filename), dpi=300)
    plt.close()
    return filename

def save_roc_curve(model_wrapper, X_test, y_test, model_name, folder):
    try:
        y_score = model_wrapper.predict_proba(X_test)
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
        plt.figure(figsize=(6, 5))
        for i in range(4):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            plt.plot(fpr, tpr, label=f'{LABEL_NAMES[i]} (AUC={auc(fpr, tpr):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('假阳性率')
        plt.ylabel('真正率')
        plt.title(f'{model_name} ROC 曲线')
        plt.legend()
        plt.tight_layout()
        filename = f"{model_name}_roc.png"
        plt.savefig(os.path.join(folder, filename), dpi=300)
        plt.close()
        return filename
    except Exception as e:
        logging.warning(f"无法生成 {model_name} 的 ROC 曲线: {e}")
        return None

def save_feature_importance(model_wrapper, feature_names, model_name, folder):
    try:
        if hasattr(model_wrapper.model, 'feature_importances_'):
            imp = model_wrapper.model.feature_importances_
            indices = np.argsort(imp)[::-1]
            plt.figure(figsize=(8, 5))
            sns.barplot(x=np.array(feature_names)[indices], y=imp[indices], palette='viridis')
            plt.xticks(rotation=45)
            plt.title(f'{model_name} 特征重要性')
            plt.tight_layout()
            filename = f"{model_name}_importance.png"
            plt.savefig(os.path.join(folder, filename), dpi=300)
            plt.close()
            return filename
    except Exception:
        pass
    return None

def save_radar_chart(metrics, model_name, folder):
    labels = np.array(['准确率\n(Accuracy)', '精确率\n(Precision)',
                       '召回率\n(Recall)', 'F1 分数\n(F1 Score)', 'Kappa系数'])
    stats = np.array([metrics['accuracy'], metrics['precision'],
                      metrics['recall'], metrics['f1'], metrics['kappa']])
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats, [stats[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, stats, color='#3498db', alpha=0.25)
    ax.plot(angles, stats, color='#2980b9', linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    plt.title(f'{model_name} 性能评估雷达图', size=14, y=1.1)
    plt.tight_layout()
    filename = f"{model_name}_radar.png"
    plt.savefig(os.path.join(folder, filename), dpi=300)
    plt.close()
    return filename

def generate_shap_beeswarm(model_obj, X_sample, feature_names, folder, model_name, class_idx=0):
    try:
        base_model = model_obj.model if hasattr(model_obj, 'model') else model_obj
        if not isinstance(base_model, (xgb.XGBClassifier, lgb.LGBMClassifier,
                                       RandomForestClassifier, cb.CatBoostClassifier)):
            return None

        X_sample = X_sample[:200] if X_sample.shape[0] > 200 else X_sample
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values_class = shap_values[class_idx]
        else:
            shap_values_class = shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_class, X_sample, feature_names=feature_names, show=False)
        plt.title(f'SHAP蜂群图 (类别: {LABEL_NAMES[class_idx]})', fontsize=14)
        plt.tight_layout()
        filename = f"{model_name}_shap_beeswarm.png"
        plt.savefig(os.path.join(folder, filename), dpi=300, bbox_inches='tight')
        plt.close()
        return filename
    except Exception as e:
        logging.error(f"SHAP 图生成失败: {e}")
        return None

# ====================== 后台训练主流程 ======================
def run_training_task(task_id='default'):
    update_status(task_id, running=True, progress=0, message='📂 加载数据并做特征工程...')
    try:
        df_raw = load_data()
        df_processed = feature_engineering(df_raw)
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler = preprocess(df_processed)
        update_status(task_id, scaler=scaler, feature_names=feature_names, progress=5)

        all_models = get_all_models(X_train.shape[1])
        results = []
        model_images_dict = {}
        best_val_f1 = -1
        best_model_obj = None
        best_model_name = None
        total_models = len(all_models)

        for idx, (name, model_wrapper) in enumerate(all_models.items()):
            progress = 5 + int((idx / total_models) * 40)
            update_status(task_id, message=f'🤖 训练 {name} ({idx+1}/{total_models})', progress=progress)
            try:
                model_copy = deepcopy(model_wrapper)
                model_copy.fit(X_train, y_train)
                y_val_pred = model_copy.predict(X_val)
                val_f1 = f1_score(y_val, y_val_pred, average='weighted')

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_obj = model_copy
                    best_model_name = name

                y_test_pred = model_copy.predict(X_test)

                cm_img = save_confusion_matrix(y_test, y_test_pred, name, app.config['RESULTS_FOLDER'])
                roc_img = save_roc_curve(model_copy, X_test, y_test, name, app.config['RESULTS_FOLDER'])
                imp_img = save_feature_importance(model_copy, feature_names, name, app.config['RESULTS_FOLDER'])

                model_images_dict[name] = {
                    'cm': cm_img,
                    'roc': roc_img,
                    'importance': imp_img
                }

                metrics = {
                    'model': name,
                    'accuracy': accuracy_score(y_test, y_test_pred),
                    'precision': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_test_pred, average='weighted'),
                    'f1': f1_score(y_test, y_test_pred, average='weighted'),
                    'kappa': cohen_kappa_score(y_test, y_test_pred)
                }
                results.append(metrics)

            except Exception as e:
                logging.error(f"模型 {name} 训练失败: {e}")

        update_status(task_id, models_performance=results, model_images=model_images_dict, progress=50)

        update_status(task_id, message='⚙️ 启动 6 种优化算法对比（XGBoost 调优）', progress=55)
        opt_results, opt_plot_path, best_opt_info = run_6_optimization_battle(X_train, y_train, task_id)

        update_status(task_id, message=f'🏆 应用最佳优化算法: {best_opt_info["opt_type"]}', progress=80)
        best_opt_model = xgb.XGBClassifier(**best_opt_info['params'])
        best_opt_model.fit(X_train, y_train)
        final_wrapper = ModelWrapper(f"XGBoost_{best_opt_info['opt_type']}", best_opt_model)

        update_status(task_id, message='📊 生成最终报告与图表...', progress=90)
        y_test_pred_final = final_wrapper.predict(X_test)
        final_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred_final),
            'precision': precision_score(y_test, y_test_pred_final, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_test_pred_final, average='weighted'),
            'f1': f1_score(y_test, y_test_pred_final, average='weighted'),
            'kappa': cohen_kappa_score(y_test, y_test_pred_final)
        }

        radar_img = save_radar_chart(final_metrics, final_wrapper.name, app.config['RESULTS_FOLDER'])
        shap_img = generate_shap_beeswarm(final_wrapper, X_train, feature_names,
                                          app.config['RESULTS_FOLDER'], "Best_Tree_Model", class_idx=0)

        update_status(task_id,
                      final_radar=radar_img,
                      shap_image=shap_img,
                      opt_comparison_img=opt_plot_path,
                      message='✅ 训练全流程完成！',
                      progress=100,
                      running=False)

    except Exception as e:
        logging.exception("训练流程发生严重错误")
        update_status(task_id, message=f'❌ 错误: {str(e)}', progress=0, running=False, error=str(e))

# ====================== Flask 路由与前端界面 ======================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def api_status():
    status = get_status()
    return jsonify({
        'running': status.get('running', False),
        'progress': status.get('progress', 0),
        'message': status.get('message', ''),
        'models_performance': status.get('models_performance'),
        'model_images': status.get('model_images'),
        'final_radar': status.get('final_radar'),
        'opt_comparison_img': status.get('opt_comparison_img'),
        'shap_image': status.get('shap_image')
    })

@app.route('/api/start_training', methods=['POST'])
def api_start_training():
    if get_status().get('running'):
        return jsonify({'error': '训练已在运行中'}), 400
    threading.Thread(target=run_training_task, args=('default',), daemon=True).start()
    return jsonify({'status': 'started'})

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

# ====================== 完整 HTML 模板 ======================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>羊只瘤胃健康智能监控平台 Pro</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css">
    <style>
        body { background: #f4f6f9; font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; }
        .dashboard-header { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); color: white;
                            padding: 1.5rem; border-radius: 0 0 20px 20px; margin-bottom: 1.5rem;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        .card { border-radius: 15px; border: none; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px; }
        .nav-tabs .nav-link { color: #495057; font-weight: bold; border-radius: 10px 10px 0 0; }
        .nav-tabs .nav-link.active { color: #203a43; border-bottom: 3px solid #203a43; }
        .chart-container { text-align: center; min-height: 300px; padding: 20px;
                           border: 1px dashed #dee2e6; border-radius: 10px; background: #fff; }
        .chart-container img { max-width: 100%; max-height: 400px; object-fit: contain; border-radius: 8px; }
    </style>
</head>
<body>
<div class="dashboard-header container-fluid">
    <div class="container d-flex justify-content-between align-items-center">
        <div>
            <h2 class="m-0"><i class="bi bi-activity"></i> 羊只瘤胃健康智能监控平台 Pro</h2>
            <small>多模型对比 · 6种优化算法 · SHAP 可解释性</small>
        </div>
        <div>
            <span id="statusBadge" class="badge bg-success px-3 py-2" style="font-size:1rem;">系统待命</span>
        </div>
    </div>
</div>

<div class="container">
    <div class="card p-3 mb-4">
        <div class="d-flex justify-content-between align-items-center flex-wrap">
            <div>
                <button id="trainBtn" class="btn btn-primary px-4 py-2">
                    <i class="bi bi-cpu"></i> 启动全流程模型训练
                </button>
            </div>
            <div style="width: 70%;">
                <div class="progress" style="height: 12px; border-radius: 10px;">
                    <div id="trainProgress" class="progress-bar progress-bar-striped progress-bar-animated bg-success" style="width: 0%"></div>
                </div>
                <div id="progressMsg" class="text-muted small mt-1 text-end">准备就绪</div>
            </div>
        </div>
    </div>

    <ul class="nav nav-tabs mb-4" id="mainTab" role="tablist">
        <li class="nav-item">
            <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab-optimization"><i class="bi bi-graph-up-arrow"></i> 优化算法对比</button>
        </li>
        <li class="nav-item">
            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-models"><i class="bi bi-table"></i> 模型性能表</button>
        </li>
        <li class="nav-item">
            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-evaluation"><i class="bi bi-bar-chart-steps"></i> 图表评估</button>
        </li>
        <li class="nav-item">
            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-shap"><i class="bi bi-diagram-3"></i> SHAP 解释</button>
        </li>
    </ul>

    <div class="tab-content">
        <div class="tab-pane fade show active" id="tab-optimization">
            <div class="card">
                <div class="card-header bg-white fw-bold">6种优化算法在 XGBoost 上的性能对比</div>
                <div class="card-body"><div id="optPlot" class="chart-container"><span class="text-muted">训练完成后显示</span></div></div>
            </div>
        </div>

        <div class="tab-pane fade" id="tab-models">
            <div class="card">
                <div class="card-header bg-white fw-bold">所有基础模型测试集性能</div>
                <div class="card-body p-0">
                    <table class="table table-hover mb-0 text-center" id="performanceTable">
                        <thead class="table-light"><tr><th>模型名称</th><th>准确率</th><th>精确率</th><th>召回率</th><th>F1分数</th><th>Kappa系数</th></tr></thead>
                        <tbody><tr><td colspan="6" class="text-muted">请先启动训练</td></tr></tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="tab-pane fade" id="tab-evaluation">
            <div class="card mb-4">
                <div class="card-header bg-white fw-bold text-primary">最终优化模型性能雷达图</div>
                <div class="card-body"><div id="radarPlaceholder" class="chart-container"><span class="text-muted">训练完成后展示</span></div></div>
            </div>

            <div class="card">
                <div class="card-header bg-white d-flex align-items-center">
                    <label class="fw-bold me-3"><i class="bi bi-funnel"></i> 切换查看模型:</label>
                    <select id="modelSelector" class="form-select w-25" onchange="switchModelCharts()"><option value="">暂无数据</option></select>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-4"><div class="card"><div class="card-header bg-white text-center">特征重要性</div><div class="card-body"><div id="chart-imp" class="chart-container"></div></div></div></div>
                        <div class="col-md-4"><div class="card"><div class="card-header bg-white text-center">混淆矩阵</div><div class="card-body"><div id="chart-cm" class="chart-container"></div></div></div></div>
                        <div class="col-md-4"><div class="card"><div class="card-header bg-white text-center">ROC 曲线</div><div class="card-body"><div id="chart-roc" class="chart-container"></div></div></div></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="tab-pane fade" id="tab-shap">
            <div class="card"><div class="card-header bg-white fw-bold"><i class="bi bi-magic"></i> 最佳树模型 SHAP 蜂群图</div><div class="card-body"><div id="shapPlaceholder" class="chart-container"><span class="text-muted">训练完成后生成</span></div></div></div>
        </div>
    </div>
</div>

<script>
    let pollingInterval = null;
    let globalImagesDict = {};

    function updateUI(data) {
        if (data.opt_comparison_img) document.getElementById('optPlot').innerHTML = `<img src="/results/${data.opt_comparison_img}?t=${Date.now()}">`;
        if (data.models_performance) {
            let tbody = '';
            data.models_performance.forEach(m => {
                tbody += `<tr><td class="fw-bold">${m.model}</td><td>${(m.accuracy*100).toFixed(2)}%</td><td>${(m.precision*100).toFixed(2)}%</td><td>${(m.recall*100).toFixed(2)}%</td><td>${(m.f1*100).toFixed(2)}%</td><td>${m.kappa.toFixed(3)}</td></tr>`;
            });
            document.querySelector('#performanceTable tbody').innerHTML = tbody;
        }
        if (data.final_radar) document.getElementById('radarPlaceholder').innerHTML = `<img src="/results/${data.final_radar}?t=${Date.now()}">`;
        if (data.shap_image) document.getElementById('shapPlaceholder').innerHTML = `<img src="/results/${data.shap_image}?t=${Date.now()}">`;
        if (data.model_images) {
            globalImagesDict = data.model_images;
            const sel = document.getElementById('modelSelector');
            sel.innerHTML = '';
            Object.keys(globalImagesDict).forEach(name => sel.innerHTML += `<option value="${name}">${name}</option>`);
            switchModelCharts();
        }
    }

    function switchModelCharts() {
        const selName = document.getElementById('modelSelector').value;
        if (!selName || !globalImagesDict[selName]) return;
        const imgs = globalImagesDict[selName], t = Date.now();
        document.getElementById('chart-imp').innerHTML = imgs.importance ? `<img src="/results/${imgs.importance}?t=${t}">` : '不支持';
        document.getElementById('chart-cm').innerHTML = imgs.cm ? `<img src="/results/${imgs.cm}?t=${t}">` : '缺失';
        document.getElementById('chart-roc').innerHTML = imgs.roc ? `<img src="/results/${imgs.roc}?t=${t}">` : '缺失';
    }

    async function pollStatus() {
        const resp = await fetch('/api/status');
        const data = await resp.json();
        document.getElementById('trainProgress').style.width = data.progress + '%';
        document.getElementById('progressMsg').innerText = data.message;
        const badge = document.getElementById('statusBadge');
        if (data.running) {
            badge.innerText = '运行中'; badge.className = 'badge bg-warning text-dark px-3 py-2';
            document.getElementById('trainBtn').disabled = true;
        } else {
            badge.innerText = '已完成'; badge.className = 'badge bg-success px-3 py-2';
            document.getElementById('trainBtn').disabled = false;
            if (data.progress === 100) { clearInterval(pollingInterval); updateUI(data); }
        }
    }

    document.getElementById('trainBtn').addEventListener('click', async () => {
        await fetch('/api/start_training', { method: 'POST' });
        pollingInterval = setInterval(pollStatus, 1000);
    });

    window.onload = () => { pollStatus(); };
</script>
</body>
</html>
'''

if __name__ == '__main__':
    print("="*50)
    print("🐏 羊只瘤胃健康智能监控平台 Pro 已启动")
    port = int(os.environ.get('PORT', 5000))          # 新增
    print(f"请在浏览器中打开: http://localhost:{port}")
    print("="*50)
    app.run(debug=False, host='0.0.0.0', port=port)   # 改动 port 参数
