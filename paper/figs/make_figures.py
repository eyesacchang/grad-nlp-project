"""Generate paper figures at ICML column width with serif font.

Output sizes:
- fig_main_p1.pdf:    text width-ish (~4.4") for 7 language pairs
- fig_geometric.pdf:  text width (~6.3") for 2 panels (intra-cos, eRank)
- fig_ablation.pdf:   text width (~6.3") for 3 panels
"""
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.axisbelow': True,
    'pdf.fonttype': 42,
})

VANILLA = '#3B7DB5'
ISO     = '#E07B27'
ACTIVE  = '#9AA4AD'
CROSSA  = '#5E6B75'

# ---------------------------------------------------------------------------
# Figure 1 (main): P@1 across 7 language pairs (3-seed mean +/- std)
# 2000-step training; significance markers computed against the larger seed-
# wise std (***: Delta > 2 sigma, **: Delta > sigma, *: Delta > sigma/2).
# ---------------------------------------------------------------------------
pairs   = ['EN-ES', 'EN-FR', 'EN-DE', 'EN-SW', 'EN-AR', 'EN-NE', 'EN-TA']
van_mu  = [0.9973, 0.9983, 0.9957, 0.8344, 0.9423, 0.9008, 0.9066]
van_sd  = [0.0010, 0.0006, 0.0006, 0.0230, 0.0204, 0.0176, 0.0133]
iso_mu  = [0.9968, 0.9988, 0.9978, 0.8689, 0.9622, 0.9154, 0.9154]
iso_sd  = [0.0002, 0.0002, 0.0013, 0.0227, 0.0132, 0.0106, 0.0192]
sig     = ['',     '*',    '**',   '**',   '*',    '*',    '']

fig, ax = plt.subplots(figsize=(4.4, 2.3))
x = np.arange(len(pairs))
w = 0.36
ax.bar(x - w/2, van_mu, w, yerr=van_sd, capsize=2.5,
       color=VANILLA, label='Vanilla ColBERT', edgecolor='white', linewidth=0.4)
ax.bar(x + w/2, iso_mu, w, yerr=iso_sd, capsize=2.5,
       color=ISO, label=r'IsoColBERT ($\lambda{=}0.5$)', edgecolor='white', linewidth=0.4)
for i, s in enumerate(sig):
    if s:
        y = max(van_mu[i], iso_mu[i]) + max(van_sd[i], iso_sd[i]) + 0.012
        ax.text(i + w/2, y, s, ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(pairs)
ax.set_ylabel('MaxSim P@1')
ax.set_ylim(0.78, 1.02)
ax.legend(loc='lower left', frameon=False)
plt.tight_layout(pad=0.2)
plt.savefig('fig_main_p1.pdf', bbox_inches='tight')
plt.close()
print('Saved fig_main_p1.pdf')

# ---------------------------------------------------------------------------
# Figure 1b (focused low-resource gains): pp gain on the 4 low-resource pairs
# ---------------------------------------------------------------------------
lr_pairs  = ['EN-SW', 'EN-AR', 'EN-NE', 'EN-TA']
lr_gains  = [3.45, 1.99, 1.46, 0.88]                      # pp gain over vanilla
# Std of the per-seed gain (computed from per-seed deltas, ddof=1)
# SW deltas: +3.09, +4.08, +3.19 -> std 0.55
# AR deltas: +1.79, +1.14, +3.04 -> std 0.97
# NE deltas: +0.69, +1.30, +2.39 -> std 0.86
# TA deltas: +1.44, +1.64, -0.45 -> std 1.15
lr_std    = [0.55, 0.97, 0.86, 1.15]                      # pp

fig, ax = plt.subplots(figsize=(3.4, 2.3))
xpos = np.arange(len(lr_pairs))
bars = ax.bar(xpos, lr_gains, 0.6, yerr=lr_std, capsize=3,
              color=ISO, edgecolor='white', linewidth=0.4)
for i, v in enumerate(lr_gains):
    ax.text(i, v + lr_std[i] + 0.08, f'+{v:.2f}', ha='center', va='bottom',
            fontsize=8, fontweight='bold', color=ISO)
ax.set_xticks(xpos)
ax.set_xticklabels(lr_pairs)
ax.set_ylabel('Gain over vanilla (pp)')
ax.set_ylim(0, 5.0)
ax.axhline(0, color='black', linewidth=0.6)
ax.set_title(r'IsoColBERT $(\lambda{=}0.5)$ gain on low-resource pairs')
plt.tight_layout(pad=0.2)
plt.savefig('fig_low_resource_gains.pdf', bbox_inches='tight')
plt.close()
print('Saved fig_low_resource_gains.pdf')

# ---------------------------------------------------------------------------
# Figure 2 (geometry): intra-cosine and effective rank, vanilla vs IsoColBERT,
# 2 panels at text width. Numbers are 3-seed means, 2000-step training.
# ---------------------------------------------------------------------------
geo_x        = np.arange(2)
geo_labels   = ['EN tokens', 'ES tokens']
van_intra    = [0.2015, 0.2937]
van_intra_sd = [0.0092, 0.0124]
iso_intra    = [0.1718, 0.1808]
iso_intra_sd = [0.0030, 0.0018]

van_erank    = [105.06, 101.73]
van_erank_sd = [0.27,    0.63]
iso_erank    = [107.50, 106.70]
iso_erank_sd = [0.31,    0.50]

fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.4))

ax = axes[0]
w = 0.36
ax.bar(geo_x - w/2, van_intra, w, yerr=van_intra_sd, capsize=2.5,
       color=VANILLA, label='Vanilla', edgecolor='white', linewidth=0.4)
ax.bar(geo_x + w/2, iso_intra, w, yerr=iso_intra_sd, capsize=2.5,
       color=ISO, label=r'IsoColBERT ($\lambda{=}0.5$)', edgecolor='white', linewidth=0.4)
ax.set_xticks(geo_x); ax.set_xticklabels(geo_labels)
ax.set_ylabel('Mean intra-cosine')
ax.set_title('Intra-cosine (lower = more isotropic)')
ax.set_ylim(0, 0.36)
ax.legend(loc='upper left', frameon=False)

ax = axes[1]
ax.bar(geo_x - w/2, van_erank, w, yerr=van_erank_sd, capsize=2.5,
       color=VANILLA, edgecolor='white', linewidth=0.4)
ax.bar(geo_x + w/2, iso_erank, w, yerr=iso_erank_sd, capsize=2.5,
       color=ISO, edgecolor='white', linewidth=0.4)
ax.set_xticks(geo_x); ax.set_xticklabels(geo_labels)
ax.set_ylabel('Effective rank')
ax.set_title('Effective rank (higher = more isotropic)')
ax.set_ylim(98, 110)

plt.tight_layout(pad=0.3)
plt.savefig('fig_geometric.pdf', bbox_inches='tight')
plt.close()
print('Saved fig_geometric.pdf')

# ---------------------------------------------------------------------------
# Figure 3 (ablation): mechanism comparison, 3 panels at text width
# ---------------------------------------------------------------------------
groups = ['Uniform', 'Active-tok.', 'Cross-attn.']
colors = [ISO, ACTIVE, CROSSA]

sw_mu  = [0.8689, 0.8482, 0.8321]
sw_sd  = [0.0227, 0.0128, 0.0334]

intra_mu = [0.1808, 0.2391, 0.2985]
intra_sd = [0.0018, 0.0101, 0.0058]

erank_mu = [106.70, 104.19, 101.90]
erank_sd = [0.50,   0.83,   0.17]

fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.0))

for ax, (mu, sd, title, ylab, ylim, fmt) in zip(axes, [
    (sw_mu,    sw_sd,    'EN-SW P@1 (higher better)',          'MaxSim P@1', (0.78, 0.92), '{:.3f}'),
    (intra_mu, intra_sd, 'Intra-cos ES (lower better)',        'Intra-cos',  (0.0, 0.36),  '{:.3f}'),
    (erank_mu, erank_sd, 'Effective rank ES (higher better)',  'eRank',      (98, 110),    '{:.1f}'),
]):
    ax.bar(range(3), mu, yerr=sd, capsize=2.5, color=colors,
           edgecolor='white', linewidth=0.4, width=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.set_ylim(ylim)
    for i, v in enumerate(mu):
        ax.text(i, v + sd[i] + (ylim[1] - ylim[0]) * 0.025,
                fmt.format(v), ha='center', va='bottom', fontsize=7, fontweight='bold')

plt.tight_layout(pad=0.3)
plt.savefig('fig_ablation.pdf', bbox_inches='tight')
plt.close()
print('Saved fig_ablation.pdf')

print('All figures generated.')
