"""Generate the three poster figures from the multi-seed numbers."""
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

FONT_DIR = os.path.join(os.path.dirname(__file__), 'fonts')
if os.path.isdir(FONT_DIR):
    for fname in os.listdir(FONT_DIR):
        if fname.lower().endswith('.ttf'):
            fm.fontManager.addfont(os.path.join(FONT_DIR, fname))

plt.rcParams.update({
    'font.family': ['Titillium Web', 'sans-serif'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.axisbelow': True,
})

VANILLA_BLUE = '#3B7DB5'
ISO_ORANGE   = '#E07B27'
ACTIVE_GREY  = '#9AA4AD'
CROSSA_GREY  = '#5E6B75'

# ---------------------------------------------------------------------------
# Figure 1: Main P@1 across 7 language pairs (batch 32, 2000 steps, 3 seeds)
# ---------------------------------------------------------------------------
pairs   = ['EN-ES', 'EN-FR', 'EN-DE', 'EN-SW', 'EN-AR', 'EN-NE', 'EN-TA']
van_mu  = [0.9973, 0.9983, 0.9957, 0.8344, 0.9423, 0.9008, 0.9066]
van_sd  = [0.0010, 0.0006, 0.0006, 0.0230, 0.0204, 0.0176, 0.0133]
iso_mu  = [0.9968, 0.9988, 0.9978, 0.8689, 0.9622, 0.9154, 0.9154]
iso_sd  = [0.0002, 0.0002, 0.0013, 0.0227, 0.0132, 0.0106, 0.0192]
sig     = ['',     '*',    '**',   '**',   '*',    '*',    '']

fig, ax = plt.subplots(figsize=(10.5, 4.6))
x = np.arange(len(pairs))
w = 0.36
ax.bar(x - w/2, van_mu, w, yerr=van_sd, capsize=4,
       color=VANILLA_BLUE, label='Vanilla ColBERT', edgecolor='white', linewidth=0.6)
ax.bar(x + w/2, iso_mu, w, yerr=iso_sd, capsize=4,
       color=ISO_ORANGE, label='IsoColBERT (λ=0.5)', edgecolor='white', linewidth=0.6)
for i, s in enumerate(sig):
    if s:
        y = max(van_mu[i], iso_mu[i]) + max(van_sd[i], iso_sd[i]) + 0.012
        ax.text(i + w/2, y, s, ha='center', va='bottom', fontsize=14, fontweight='bold', color='#222')
ax.set_xticks(x); ax.set_xticklabels(pairs, fontsize=13)
ax.set_ylabel('MaxSim P@1', fontsize=13)
ax.set_ylim(0.78, 1.02)
ax.set_title('Cross-lingual retrieval (FLORES+, 3 seeds, batch 32, 2000 steps)', fontsize=13, pad=10)
ax.legend(loc='lower left', fontsize=11, frameon=False)
ax.tick_params(axis='y', labelsize=11)
plt.tight_layout()
plt.savefig('fig_main_p1.pdf', bbox_inches='tight')
plt.close()
print('Saved fig_main_p1.pdf')

# ---------------------------------------------------------------------------
# Figure 2: Geometric metrics, side-by-side vanilla vs IsoColBERT
# ---------------------------------------------------------------------------
van_ic  = [0.2015, 0.2937];  iso_ic  = [0.1718, 0.1808]
van_er  = [105.06, 101.73];  iso_er  = [107.50, 106.70]
van_ic_sd = [0.0092, 0.0124];  iso_ic_sd = [0.0030, 0.0018]
van_er_sd = [0.27,    0.63];   iso_er_sd = [0.31,    0.50]

fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.6))
x = np.arange(2); w = 0.36

# Left: intra-cosine side by side
ax = axes[0]
ax.bar(x - w/2, van_ic, w, yerr=van_ic_sd, capsize=4,
       color=VANILLA_BLUE, label='Vanilla ColBERT', edgecolor='white', linewidth=0.6)
ax.bar(x + w/2, iso_ic, w, yerr=iso_ic_sd, capsize=4,
       color=ISO_ORANGE, label='IsoColBERT (λ=0.5)', edgecolor='white', linewidth=0.6)
ax.set_xticks(x); ax.set_xticklabels(['EN tokens', 'ES tokens'], fontsize=15)
ax.set_ylabel('Mean intra-cosine', fontsize=15)
ax.set_ylim(0, 0.42)
ax.legend(loc='upper left', fontsize=13, frameon=False)
ax.tick_params(axis='y', labelsize=12)

# Right: eRank side by side
ax = axes[1]
ax.bar(x - w/2, van_er, w, yerr=van_er_sd, capsize=4,
       color=VANILLA_BLUE, edgecolor='white', linewidth=0.6)
ax.bar(x + w/2, iso_er, w, yerr=iso_er_sd, capsize=4,
       color=ISO_ORANGE, edgecolor='white', linewidth=0.6)
ax.set_xticks(x); ax.set_xticklabels(['EN tokens', 'ES tokens'], fontsize=15)
ax.set_ylabel('Effective rank', fontsize=15)
ax.set_ylim(96, 112)
ax.tick_params(axis='y', labelsize=12)

plt.suptitle('Isotropy Regularization Reshapes Token Embedding Geometry',
             fontsize=17, y=1.02)
plt.tight_layout()
plt.savefig('fig_geometric.pdf', bbox_inches='tight')
plt.close()
print('Saved fig_geometric.pdf')

# ---------------------------------------------------------------------------
# Figure 3: 3-way ablation on EN-SW + intra-cos ES + eRank ES
# (batch 32, 2000 steps, 3 seeds)
# ---------------------------------------------------------------------------
groups = ['Uniform\n(IsoColBERT)', 'Active-token', 'Cross-attention']
colors = [ISO_ORANGE, ACTIVE_GREY, CROSSA_GREY]

sw_mu  = [0.8689, 0.8482, 0.8321]
sw_sd  = [0.0227, 0.0128, 0.0334]

intra_mu = [0.1808, 0.2391, 0.2985]
intra_sd = [0.0018, 0.0101, 0.0058]

erank_mu = [106.70, 104.19, 101.90]
erank_sd = [0.50,   0.83,   0.17]

fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.4))

for ax, (mu, sd, title, ylab, ylim, fmt) in zip(axes, [
    (sw_mu,    sw_sd,    r'EN-SW P@1 (higher better)',          'MaxSim P@1',     (0.78, 0.92), '{:.4f}'),
    (intra_mu, intra_sd, r'Intra-cos ES (lower better)',        'Intra-cos',      (0.0, 0.36),  '{:.4f}'),
    (erank_mu, erank_sd, r'Effective rank ES (higher better)',  'eRank',          (98, 110),    '{:.2f}'),
]):
    bars = ax.bar(range(3), mu, yerr=sd, capsize=4, color=colors,
                  edgecolor='white', linewidth=0.6, width=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel(ylab, fontsize=12)
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_ylim(ylim)
    ax.tick_params(axis='y', labelsize=10)
    for i, v in enumerate(mu):
        ax.text(i, v + (sd[i] + (ylim[1]-ylim[0]) * 0.02),
                fmt.format(v), ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Mechanism ablations: localized and object-level constraints underperform',
             fontsize=13, y=1.04)
plt.tight_layout()
plt.savefig('fig_ablation.pdf', bbox_inches='tight')
plt.close()
print('Saved fig_ablation.pdf')

print('All figures generated.')
