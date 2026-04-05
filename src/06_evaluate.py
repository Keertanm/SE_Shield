import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CM_DIR      = os.path.join(RESULTS_DIR, 'confusion_matrices')
ROC_DIR     = os.path.join(RESULTS_DIR, 'roc_curves')
os.makedirs(CM_DIR,  exist_ok=True)
os.makedirs(ROC_DIR, exist_ok=True)

# ── Load results ──────────────────────────────────────────────────────────────
print("=" * 60)
print("LOADING RESULTS")
print("=" * 60)

summary     = joblib.load(os.path.join(MODELS_DIR, 'results_summary.pkl'))
winner_name = summary['winner_name']
results     = summary['results']
y_test      = summary['y_test']
all_y_pred  = summary['all_y_pred']
all_y_prob  = summary['all_y_prob']

print(f"Winner : {winner_name}")
print(f"Models : {list(results.keys())}")

MODEL_COLORS = {
    'Logistic Regression' : '#3b82f6',
    'Random Forest'       : '#10b981',
    'XGBoost'             : '#f59e0b',
    'SVM'                 : '#ef4444',
    'Naive Bayes'         : '#8b5cf6',
}

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — METRICS COMPARISON BAR CHART
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PLOT 1 — METRICS COMPARISON")
print("=" * 60)

metrics      = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
model_names  = list(results.keys())

fig, ax = plt.subplots(figsize=(14, 7))

x     = np.arange(len(metrics))
width = 0.15
offsets = np.linspace(-(len(model_names)-1)/2, (len(model_names)-1)/2, len(model_names))

for i, (name, offset) in enumerate(zip(model_names, offsets)):
    values = [results[name][m] * 100 for m in metrics]
    bars   = ax.bar(x + offset * width, values, width,
                    label=name, color=MODEL_COLORS[name],
                    edgecolor='black', linewidth=0.5, alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.05,
                f'{val:.1f}', ha='center', va='bottom',
                fontsize=6.5, rotation=90)

ax.set_xlabel('Metric', fontsize=13)
ax.set_ylabel('Score (%)', fontsize=13)
ax.set_title('Model Comparison — All Metrics', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=12)
ax.set_ylim(90, 102)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'metrics_comparison.png'), dpi=150)
plt.show()
print("Saved → results/metrics_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — CONFUSION MATRICES (all 5 in one figure)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PLOT 2 — CONFUSION MATRICES")
print("=" * 60)

fig, axes = plt.subplots(1, 5, figsize=(22, 5))

for ax, (name, y_pred) in zip(axes, all_y_pred.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', ax=ax,
        cmap='Blues', linewidths=0.5,
        xticklabels=['Legit', 'Attack'],
        yticklabels=['Legit', 'Attack'],
        annot_kws={'size': 13}
    )
    winner_tag = " 🏆" if name == winner_name else ""
    ax.set_title(f'{name}{winner_tag}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual',    fontsize=10)

    tn, fp, fn, tp = cm.ravel()
    ax.set_xlabel(
        f'Predicted\nTP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}',
        fontsize=9
    )

plt.suptitle('Confusion Matrices — All 5 Models', fontsize=14,
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrices.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved → results/confusion_matrices.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — ROC CURVES (all 5 on one chart)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PLOT 3 — ROC CURVES")
print("=" * 60)

fig, ax = plt.subplots(figsize=(9, 7))

for name, y_prob in all_y_prob.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc     = auc(fpr, tpr)
    lw          = 3 if name == winner_name else 1.5
    ls          = '-' if name == winner_name else '--'
    winner_tag  = " 🏆" if name == winner_name else ""
    ax.plot(fpr, tpr, color=MODEL_COLORS[name],
            lw=lw, ls=ls,
            label=f'{name}{winner_tag} (AUC={roc_auc*100:.2f}%)')

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate',  fontsize=13)
ax.set_title('ROC Curves — All 5 Models', fontsize=15, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(ROC_DIR, 'roc_curves.png'), dpi=150)
plt.show()
print("Saved → results/roc_curves/roc_curves.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — INFERENCE SPEED vs F1 SCORE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PLOT 4 — SPEED vs ACCURACY TRADEOFF")
print("=" * 60)

fig, ax = plt.subplots(figsize=(9, 6))

for name, r in results.items():
    x_val  = r['infer_ms'] + 0.0001
    y_val  = r['f1_score'] * 100
    color  = MODEL_COLORS[name]
    marker = '*' if name == winner_name else 'o'
    size   = 300 if name == winner_name else 150

    ax.scatter(x_val, y_val, color=color,
               s=size, marker=marker,
               edgecolors='black', linewidth=1.5,
               zorder=5, label=name)
    ax.annotate(
        name, (x_val, y_val),
        textcoords='offset points',
        xytext=(8, 4), fontsize=9
    )

ax.axvline(x=5.0, color='red', linestyle='--',
           linewidth=1.5, label='Stage 1 limit (5ms)')
ax.set_xlabel('Inference Time per Sample (ms)', fontsize=12)
ax.set_ylabel('F1 Score (%)', fontsize=12)
ax.set_title('Speed vs Accuracy Tradeoff\n(all models meet Stage 1 <5ms requirement)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'speed_vs_accuracy.png'), dpi=150)
plt.show()
print("Saved → results/speed_vs_accuracy.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — CV F1 WITH ERROR BARS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PLOT 5 — CROSS VALIDATION F1 WITH ERROR BARS")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

names    = list(results.keys())
cv_f1s   = [results[n]['cv_f1'] * 100    for n in names]
cv_stds  = [results[n]['cv_f1_std'] * 100 for n in names]
colors   = [MODEL_COLORS[n] for n in names]

bars = ax.bar(names, cv_f1s, color=colors,
              edgecolor='black', linewidth=0.8, alpha=0.85)
ax.errorbar(names, cv_f1s, yerr=cv_stds,
            fmt='none', color='black',
            capsize=6, capthick=2, elinewidth=2)

for bar, val, std in zip(bars, cv_f1s, cv_stds):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + std + 0.05,
            f'{val:.2f}%\n±{std:.2f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

winner_idx = names.index(winner_name)
bars[winner_idx].set_edgecolor('gold')
bars[winner_idx].set_linewidth(3)
ax.text(winner_idx, cv_f1s[winner_idx] + cv_stds[winner_idx] + 0.8,
        '🏆', ha='center', fontsize=16)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('CV F1 Score (%)', fontsize=12)
ax.set_title('5-Fold Cross Validation F1 Scores with Standard Deviation',
             fontsize=13, fontweight='bold')
ax.set_ylim(93, 101)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=15, ha='right')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'cv_f1_comparison.png'), dpi=150)
plt.show()
print("Saved → results/cv_f1_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PRINT FINAL CLASSIFICATION REPORT FOR WINNER
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"CLASSIFICATION REPORT — {winner_name} (WINNER)")
print("=" * 60)

print(classification_report(
    y_test,
    all_y_pred[winner_name],
    target_names=['Legitimate', 'Attack']
))

print("\n" + "=" * 60)
print("ALL PLOTS SAVED")
print("=" * 60)
print("""
results/
├── metrics_comparison.png     ← bar chart all metrics
├── confusion_matrices.png     ← all 5 confusion matrices
├── speed_vs_accuracy.png      ← speed vs F1 tradeoff
├── cv_f1_comparison.png       ← CV F1 with error bars
└── roc_curves/
    └── roc_curves.png         ← all 5 ROC curves
""")
print("Evaluation complete. Stage 1 pipeline finished.")