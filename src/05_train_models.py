import os
import joblib
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("  [INFO] tqdm not installed — run 'pip install tqdm' for progress bars")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(BASE_DIR, 'data', 'splits')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Load TF-IDF features ──────────────────────────────────────────────────────
print("=" * 65)
print("LOADING TF-IDF FEATURES — FIXED ACROSS ALL EXPERIMENTS")
print("=" * 65)

X_train = joblib.load(os.path.join(SPLITS_DIR, 'X_train_tfidf.pkl'))
X_test  = joblib.load(os.path.join(SPLITS_DIR, 'X_test_tfidf.pkl'))
y_train = joblib.load(os.path.join(SPLITS_DIR, 'y_train.pkl'))
y_test  = joblib.load(os.path.join(SPLITS_DIR, 'y_test.pkl'))

X_train = X_train.tocsr()
X_test  = X_test.tocsr()

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_train : Attack={y_train.sum():,}  Legit={(y_train==0).sum():,}")
print(f"y_test  : Attack={y_test.sum():,}   Legit={(y_test==0).sum():,}")
print("\nFeatures fixed — only tuning strategies differ across experiments.")


# ── tqdm-compatible GridSearchCV wrapper ──────────────────────────────────────
class ProgressGridSearchCV(GridSearchCV):
    """
    GridSearchCV subclass that shows a tqdm progress bar over all
    candidate x fold combinations. Falls back silently without tqdm.
    """
    def fit(self, X, y=None, **fit_params):
        if not TQDM_AVAILABLE:
            return super().fit(X, y, **fit_params)

        total = (
            len(self.param_grid) if isinstance(self.param_grid, list)
            else np.prod([len(v) for v in self.param_grid.values()])
        )
        n_splits   = self.cv.get_n_splits() if hasattr(self.cv, 'get_n_splits') else 5
        total_fits = int(total * n_splits)

        with tqdm(total=total_fits, desc="  Grid search", unit="fit",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} fits [{elapsed}<{remaining}]") as pbar:
            self.set_params(verbose=0)
            result = super().fit(X, y, **fit_params)
            pbar.update(total_fits - pbar.n)
        return result


# ===============================================================================
# SHARED EVALUATION FUNCTION
# ===============================================================================
def evaluate(name: str, model, X_te, y_te) -> dict:
    t0     = time.time()
    y_pred = model.predict(X_te)
    infer  = (time.time() - t0) * 1000 / len(y_te)

    try:
        y_prob = model.predict_proba(X_te)[:, 1]
        auc    = roc_auc_score(y_te, y_prob)
    except Exception:
        y_prob = y_pred.astype(float)
        auc    = roc_auc_score(y_te, y_prob)

    acc  = accuracy_score (y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='macro', zero_division=0)
    rec  = recall_score   (y_te, y_pred, average='macro', zero_division=0)
    f1   = f1_score       (y_te, y_pred, average='macro', zero_division=0)
    cm   = confusion_matrix(y_te, y_pred)

    print(f"\n{'─'*65}")
    print(f"  RESULTS — {name}")
    print(f"{'─'*65}")
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"  Precision  : {prec*100:.2f}%  (macro)")
    print(f"  Recall     : {rec*100:.2f}%   (macro)")
    print(f"  F1 Score   : {f1*100:.2f}%   (macro)")
    print(f"  ROC-AUC    : {auc*100:.2f}%")
    print(f"  Infer/samp : {infer:.4f} ms")
    print(f"\n  Confusion Matrix:")
    tn, fp, fn, tp = cm.ravel()
    print(f"    TP={tp:,}  FP={fp:,}")
    print(f"    FN={fn:,}  TN={tn:,}")
    print(f"\n  Classification Report:")
    print(classification_report(y_te, y_pred, target_names=['Legitimate', 'Attack']))

    return {
        'name'      : name,
        'accuracy'  : acc,
        'precision' : prec,
        'recall'    : rec,
        'f1_score'  : f1,
        'roc_auc'   : auc,
        'infer_ms'  : infer,
        'y_pred'    : y_pred,
        'y_prob'    : y_prob,
    }


# ===============================================================================
# EXPERIMENT 1 — PAPER 10 BASELINE
# LinearSVC, no tuning, default parameters
# ===============================================================================
def experiment_paper10(X_tr, y_tr, X_te, y_te) -> dict:
    """
    Paper 10 — Mental Health Study
    LinearSVC with default parameters, no tuning.
    Serves as the baseline for all comparisons.
    """
    print("\n" + "=" * 65)
    print("EXPERIMENT 1 — PAPER 10 (Baseline LinearSVC, No Tuning)")
    print("=" * 65)
    print("  Model  : LinearSVC (default params)")
    print("  Tuning : NONE")
    print("  Note   : This is the baseline all others are compared against")

    model = CalibratedClassifierCV(
        LinearSVC(dual='auto', max_iter=3000, random_state=42)
    )

    if TQDM_AVAILABLE:
        with tqdm(total=1, desc="  Training", unit="model",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
            t0 = time.time()
            model.fit(X_tr, y_tr)
            train_time = (time.time() - t0) * 1000
            pbar.update(1)
    else:
        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_time = (time.time() - t0) * 1000

    print(f"\n  Train time : {train_time:.0f} ms")

    result = evaluate("Paper 10 — Baseline LinearSVC", model, X_te, y_te)
    result['train_time_ms'] = train_time
    result['best_params']   = "Default — no tuning"

    joblib.dump(model, os.path.join(MODELS_DIR, 'svm_paper10_baseline.pkl'))
    print("  Saved → models/svm_paper10_baseline.pkl")
    return result


# ===============================================================================
# EXPERIMENT 2 — ENRON PAPER 2
# SVC + GridSearch over C and kernel, scoring=f1_macro
# ===============================================================================
def experiment_enron2(X_tr, y_tr, X_te, y_te) -> dict:
    """
    Enron Paper 2 — 98.7% accuracy paper
    SVC with GridSearchCV over C and kernel.
    Scoring: f1_macro  |  CV folds: 5 stratified
    """
    print("\n" + "=" * 65)
    print("EXPERIMENT 2 — ENRON PAPER 2 (GridSearch C + Kernel)")
    print("=" * 65)

    param_grid   = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    n_candidates = len(param_grid['C']) * len(param_grid['kernel'])
    n_splits     = 5

    print(f"  Model      : SVC")
    print(f"  Tuning     : GridSearchCV")
    print(f"  Param grid : {param_grid}")
    print(f"  Scoring    : f1_macro")
    print(f"  CV folds   : {n_splits} (stratified)")
    print(f"  Total fits : {n_candidates} candidates x {n_splits} folds = {n_candidates * n_splits}")

    base_svc = SVC(probability=True, random_state=42, max_iter=3000)
    cv       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    grid = ProgressGridSearchCV(
        base_svc, param_grid,
        scoring = 'f1_macro',
        cv      = cv,
        n_jobs  = -1,
        verbose = 0,
        refit   = True,
    )

    t0 = time.time()
    grid.fit(X_tr, y_tr)
    train_time = (time.time() - t0) * 1000

    print(f"\n  Best params : {grid.best_params_}")
    print(f"  Best CV F1  : {grid.best_score_*100:.2f}%")
    print(f"  Train time  : {train_time/1000:.1f}s")

    result = evaluate("Enron Paper 2 — GridSearch C+Kernel",
                      grid.best_estimator_, X_te, y_te)
    result['train_time_ms'] = train_time
    result['best_params']   = str(grid.best_params_)
    result['cv_best_score'] = grid.best_score_

    joblib.dump(grid.best_estimator_, os.path.join(MODELS_DIR, 'svm_enron2.pkl'))
    print("  Saved → models/svm_enron2.pkl")
    return result


# ===============================================================================
# EXPERIMENT 3 — ENRON PAPER 1
# SVC RBF + GridSearch over C and gamma
# ===============================================================================
def experiment_enron1(X_tr, y_tr, X_te, y_te) -> dict:
    """
    Enron Paper 1
    SVC with RBF kernel fixed.
    GridSearchCV over C and gamma only.
    Scoring: f1_macro  |  CV folds: 5 stratified
    """
    print("\n" + "=" * 65)
    print("EXPERIMENT 3 — ENRON PAPER 1 (RBF + GridSearch C + Gamma)")
    print("=" * 65)

    param_grid   = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]}
    n_candidates = len(param_grid['C']) * len(param_grid['gamma'])
    n_splits     = 5

    print(f"  Model      : SVC (kernel=rbf, FIXED)")
    print(f"  Tuning     : GridSearchCV")
    print(f"  Param grid : {param_grid}")
    print(f"  Scoring    : f1_macro")
    print(f"  CV folds   : {n_splits} (stratified)")
    print(f"  Total fits : {n_candidates} candidates x {n_splits} folds = {n_candidates * n_splits}")

    base_svc = SVC(kernel='rbf', probability=True, random_state=42, max_iter=3000)
    cv       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    grid = ProgressGridSearchCV(
        base_svc, param_grid,
        scoring = 'f1_macro',
        cv      = cv,
        n_jobs  = -1,
        verbose = 0,
        refit   = True,
    )

    t0 = time.time()
    grid.fit(X_tr, y_tr)
    train_time = (time.time() - t0) * 1000

    print(f"\n  Best params : {grid.best_params_}")
    print(f"  Best CV F1  : {grid.best_score_*100:.2f}%")
    print(f"  Train time  : {train_time/1000:.1f}s")

    result = evaluate("Enron Paper 1 — RBF GridSearch C+Gamma",
                      grid.best_estimator_, X_te, y_te)
    result['train_time_ms'] = train_time
    result['best_params']   = str(grid.best_params_)
    result['cv_best_score'] = grid.best_score_

    joblib.dump(grid.best_estimator_, os.path.join(MODELS_DIR, 'svm_enron1.pkl'))
    print("  Saved → models/svm_enron1.pkl")
    return result


# ===============================================================================
# MAIN — RUN ALL EXPERIMENTS
# ===============================================================================
if __name__ == '__main__':

    print("\n" + "=" * 65)
    print("RUNNING ALL 3 SVM TUNING EXPERIMENTS")
    print("Fixed   : TF-IDF features, same dataset, same train/test split")
    print("Variable: tuning strategy only")
    print("=" * 65)

    total_start = time.time()
    results     = []

    experiments = [
        ("Exp 1: Paper 10 Baseline", experiment_paper10,
         dict(X_tr=X_train, y_tr=y_train, X_te=X_test, y_te=y_test)),
        ("Exp 2: Enron Paper 2",     experiment_enron2,
         dict(X_tr=X_train, y_tr=y_train, X_te=X_test, y_te=y_test)),
        ("Exp 3: Enron Paper 1",     experiment_enron1,
         dict(X_tr=X_train, y_tr=y_train, X_te=X_test, y_te=y_test)),
    ]

    if TQDM_AVAILABLE:
        exp_bar = tqdm(
            experiments,
            desc="Overall progress",
            unit="exp",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} experiments [{elapsed}<{remaining}]"
        )
        for label, fn, kwargs in exp_bar:
            exp_bar.set_description(f"Running {label}")
            results.append(fn(**kwargs))
    else:
        for label, fn, kwargs in experiments:
            results.append(fn(**kwargs))

    total_time = (time.time() - total_start) / 60

    # ── Final comparison table ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL COMPARISON TABLE — ALL SVM TUNING STRATEGIES")
    print("=" * 65)

    baseline_f1 = results[0]['f1_score']

    print(f"\n{'Model':<45} {'Acc':>7} {'Prec':>7} {'Rec':>7} "
          f"{'F1':>7} {'AUC':>7} {'vs Base':>8}")
    print("-" * 95)

    for r in results:
        delta     = r['f1_score'] - baseline_f1
        delta_str = f"+{delta*100:.2f}%" if delta >= 0 else f"{delta*100:.2f}%"
        best_mark = " <-BEST" if r['f1_score'] == max(x['f1_score'] for x in results) else ""
        print(
            f"{r['name']:<45} "
            f"{r['accuracy']*100:>6.2f}% "
            f"{r['precision']*100:>6.2f}% "
            f"{r['recall']*100:>6.2f}% "
            f"{r['f1_score']*100:>6.2f}% "
            f"{r['roc_auc']*100:>6.2f}% "
            f"{delta_str:>8}"
            f"{best_mark}"
        )

    # ── Best params summary ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("BEST HYPERPARAMETERS PER EXPERIMENT")
    print("=" * 65)
    for r in results:
        print(f"\n  {r['name']}")
        print(f"    {r['best_params']}")

    # ── Analysis ───────────────────────────────────────────────────────────────
    best    = max(results, key=lambda x: x['f1_score'])
    worst   = min(results, key=lambda x: x['f1_score'])
    fastest = min(results, key=lambda x: x['train_time_ms'])

    print("\n" + "=" * 65)
    print("ANALYSIS")
    print("=" * 65)
    print(f"""
  Best tuning strategy  : {best['name']}
  Best F1 score         : {best['f1_score']*100:.2f}%
  Improvement vs base   : +{(best['f1_score']-baseline_f1)*100:.2f}%

  Weakest strategy      : {worst['name']}
  Weakest F1 score      : {worst['f1_score']*100:.2f}%

  Fastest to train      : {fastest['name']}
  Fastest train time    : {fastest['train_time_ms']/1000:.1f}s

  Total experiment time : {total_time:.1f} minutes

  Trade-off summary:
  - Paper 10 (baseline) : fastest, least tuned, lowest F1
  - Enron Paper 2       : moderate tuning, good speed/accuracy balance
  - Enron Paper 1       : RBF focused, broad gamma search
    """)

    # ── Save all results ───────────────────────────────────────────────────────
    joblib.dump({
        'results'    : [{k: v for k, v in r.items()
                         if k not in ['y_pred', 'y_prob']}
                        for r in results],
        'y_test'     : y_test,
        'all_y_pred' : {r['name']: r['y_pred'] for r in results},
        'all_y_prob' : {r['name']: r['y_prob'] for r in results},
        'winner'     : best['name'],
    }, os.path.join(MODELS_DIR, 'svm_comparison_results.pkl'))

    print("Saved -> models/svm_comparison_results.pkl")
    print("\nAll experiments complete.")