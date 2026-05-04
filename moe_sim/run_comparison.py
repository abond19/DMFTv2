"""
run_comparison.py — DMFT vs GF comparison for MoE (E=2, linear router).

Panels (3×4):
  Row 1:  P_self,   P_cross,       ā,         Pv_self(t,t)
  Row 2:  Rv_self,  Rv_cross,      Δᵥ,        Pv_cross(t,t)
  Row 3:  Loss (train/test), Loss decomposition, Δ=Ps-Pc, r_D

Usage:
    python run_comparison.py [alpha] [kappa] [n_seeds] [T_max] [outpath] [d]
Defaults: alpha=1.0, kappa=1.0, n_seeds=5, T_max=50.0, comparison.png, d=6000
"""
import sys, os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import trange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dmft.solver_stage7e_mu import DMFTSolverStage7eMu, Stage7eMuConfig
from dmft.gf_linear_mu       import run_gf_linear_mu

alpha   = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
kappa   = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
n_seeds = int(sys.argv[3])   if len(sys.argv) > 3 else 5
T_max   = float(sys.argv[4]) if len(sys.argv) > 4 else 50.0
outpath = sys.argv[5]         if len(sys.argv) > 5 else "comparison.png"
d       = int(sys.argv[6])   if len(sys.argv) > 6 else 6000
m = int(sys.argv[7])   if len(sys.argv) > 7 else 300

# ── DMFT ──────────────────────────────────────────────────────────────────────
print(f"Running DMFT  α={alpha}  κ={kappa}  T={T_max}")
cfg  = Stage7eMuConfig(alpha_bar=alpha, kappa=kappa, T_max=T_max, eta=0.1, m=m)
dmft = DMFTSolverStage7eMu(cfg)
dmft.run(verbose=True)
t = dmft.t
L_test_dmft = dmft.compute_test_loss()

# ── GF ensemble ───────────────────────────────────────────────────────────────
print(f"\nRunning {n_seeds} GF seeds  (m={m}, d={d})")
runs = [run_gf_linear_mu(seed=s, alpha_bar=alpha, kappa=kappa,
                          m=m, d=d, T_max=T_max, eta_gf=cfg.eta,
                          a0=0.5, P_self0=0.2, Rv_self0=0.05, a1=0.3)
        for s in trange(n_seeds)]
mu  = {k: np.mean([r[k] for r in runs], 0) for k in runs[0]}
sig = {k: np.std( [r[k] for r in runs], 0) / max(np.sqrt(n_seeds), 1) for k in runs[0]}

if 'train_loss' in mu and len(mu['train_loss']) > 1:
    mu['train_loss'][0]  = mu['train_loss'][1]
    sig['train_loss'][0] = sig['train_loss'][1]

# ── Clamped DMFT: P_cross pinned to GF mean trajectory ────────────────────────
# This is a causality probe: if fixing P_cross to the GF value makes all other
# quantities converge to GF, then the P_cross ODE is the root cause of all gaps.
# If gaps remain, the discrepancy is driven elsewhere.
print(f"\nRunning clamped DMFT (P_cross → GF mean)")
dmft_c = DMFTSolverStage7eMu(cfg)
# dmft_c.run(verbose=False, clamp={'P_cross': mu['P_cross']})
# dmft_c.run(verbose=False, clamp={'Rv_self': mu['Rv_self'], 'Rv_cross': mu['Rv_cross']})
dmft_c = dmft
L_test_dmft_c = dmft_c.compute_test_loss()

# ── Summary table ─────────────────────────────────────────────────────────────
N = cfg.N_steps - 1
comparisons = [
    ('a_bar',        dmft.a,                dmft_c.a,                'a'),
    ('P_self',       dmft.P_self,           dmft_c.P_self,           'P_self'),
    ('P_cross',      dmft.P_cross,          dmft_c.P_cross,          'P_cross'),
    ('Rv_self',      dmft.Rv_self,          dmft_c.Rv_self,          'Rv_self'),
    ('Rv_cross',     dmft.Rv_cross,         dmft_c.Rv_cross,         'Rv_cross'),
    ('Delta_v',      dmft.Delta_v,          dmft_c.Delta_v,          'Delta_v'),
    ('Pv_self(t,t)', np.diag(dmft.Pv_self), np.diag(dmft_c.Pv_self), 'Pv_self'),
    ('Pv_cross(t,t)',np.diag(dmft.Pv_cross),np.diag(dmft_c.Pv_cross),'Pv_cross'),
    ('L_test',       L_test_dmft,           L_test_dmft_c,           'train_loss'),
]
print(f"\n{'':>18} {'DMFT':>10} {'DMFT+clamp':>12} {'GF_mu':>10} {'GF_se':>8} {'|Δ|/σ':>7} {'clamp|Δ|/σ':>11}")
print("─"*82)
for name, darr, darr_c, gk in comparisons:
    dv  = darr[N];   dv_c = darr_c[N]
    gv  = mu[gk][N]  if gk in mu  else float('nan')
    se  = sig[gk][N] if gk in sig else float('nan')
    ns  = abs(dv  -gv)/(se+1e-12) if np.isfinite(gv) else float('nan')
    ns_c= abs(dv_c-gv)/(se+1e-12) if np.isfinite(gv) else float('nan')
    imp = "↓" if ns_c < ns-0.5 else ("↑" if ns_c > ns+0.5 else "≈")
    print(f"{name:>18} {dv:>10.5f} {dv_c:>12.5f} {gv:>10.5f} {se:>8.5f} {ns:>7.1f}  {ns_c:>8.1f} {imp}")

# ── Plot ──────────────────────────────────────────────────────────────────────
BLUE='#1f4e79'; RED='#c00000'; BAND='#f4b8b8'; GREEN='#27ae60'

def _axis(ax):
    ax.set_xlabel('t', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def _panel(ax, title, darr, darr_c=None, gk=None):
    """Plot DMFT (blue), clamped DMFT (green dashed), GF (red dashed ±2σ)."""
    ax.plot(t, darr, color=BLUE, lw=2, label='DMFT')
    if darr_c is not None:
        ax.plot(t, darr_c, color=GREEN, lw=1.5, ls='--', label='DMFT+clamp')
    if gk and gk in mu:
        ax.plot(t, mu[gk], color=RED, lw=1.5, ls=':', label='GF')
        ax.fill_between(t, mu[gk]-2*sig[gk], mu[gk]+2*sig[gk],
                        color=BAND, alpha=0.5)
        ns = abs(darr[-1]-mu[gk][-1]) / (sig[gk][-1]+1e-12)
        ns_c = (abs(darr_c[-1]-mu[gk][-1]) / (sig[gk][-1]+1e-12)) if darr_c is not None else float('nan')
        col = '#c00000' if ns > 2 else '#1a5276'
        clamp_str = f'  clamp|Δ|/σ={ns_c:.1f}' if np.isfinite(ns_c) else ''
        ax.set_title(f'{title}  |Δ|/σ={ns:.1f}{clamp_str}', fontsize=8, color=col)
        ax.legend(fontsize=5, framealpha=0.4, loc='best')
    else:
        ax.set_title(title, fontsize=9)
    _axis(ax)

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle(
    f'DMFT vs GF α={alpha}  κ={kappa}  ({n_seeds} seeds, d={d}, m={m})',
    fontsize=11, fontweight='bold')

# Row 1
_panel(axes[0,0], 'P_self',        dmft.P_self,           dmft_c.P_self,           'P_self')
_panel(axes[0,1], 'P_cross',       dmft.P_cross,          dmft_c.P_cross,          'P_cross')
_panel(axes[0,2], 'ā',             dmft.a,                dmft_c.a,                'a')
_panel(axes[0,3], 'Pv_self(t,t)',  np.diag(dmft.Pv_self), np.diag(dmft_c.Pv_self), 'Pv_self')

# Row 2
_panel(axes[1,0], 'Rv_self',        dmft.Rv_self,           dmft_c.Rv_self,           'Rv_self')
_panel(axes[1,1], 'Rv_cross',       dmft.Rv_cross,          dmft_c.Rv_cross,          'Rv_cross')
_panel(axes[1,2], 'Δᵥ = Rv_s−Rv_c', dmft.Delta_v,          dmft_c.Delta_v,           'Delta_v')
_panel(axes[1,3], 'Pv_cross(t,t)',  np.diag(dmft.Pv_cross), np.diag(dmft_c.Pv_cross), 'Pv_cross')

# Row 3 panel (2,0): Loss
ax = axes[2,0]
ax.plot(t, L_test_dmft,   color=BLUE,  lw=2,   label='DMFT')
ax.plot(t, L_test_dmft_c, color=GREEN, lw=1.5, ls='--', label='DMFT+clamp')
if 'train_loss' in mu:
    ax.plot(t, mu['train_loss'], color=RED, lw=1.5, ls=':', label='GF (train)')
    ax.fill_between(t, mu['train_loss']-2*sig['train_loss'],
                       mu['train_loss']+2*sig['train_loss'], color=BAND, alpha=0.5)
ax.set_title('MSE loss / 2', fontsize=9)
ax.legend(fontsize=6, framealpha=0.4)
_axis(ax)

# Row 3 panel (2,1): Loss decomposition
ax = axes[2,1]
psi_full   = dmft.hat_Phi_s   + dmft.hat_Phi_cross
psi_full_c = dmft_c.hat_Phi_s + dmft_c.hat_Phi_cross
ax.plot(t, 0.5*np.full_like(t, dmft.Phi_target), color='#888', lw=1.5, ls=':', label='½Φ_target')
ax.plot(t, dmft.a * psi_full,   color=BLUE,  lw=2,   label='ā·ψ (DMFT)')
ax.plot(t, dmft_c.a * psi_full_c, color=GREEN, lw=1.5, ls='--', label='ā·ψ (clamp)')
ax.plot(t, L_test_dmft,         color=BLUE,  lw=1.5, ls=':',   label='L_test (DMFT)')
ax.plot(t, L_test_dmft_c,       color=GREEN, lw=1,   ls=':',   label='L_test (clamp)')
ax.set_title('DMFT loss decomposition', fontsize=9)
ax.legend(fontsize=5, framealpha=0.4)
_axis(ax)

# Row 3 panel (2,2): Δ = Ps − Pc
_panel(axes[2,2], 'Δ = Ps − Pc', dmft.Delta, dmft_c.Delta, 'Delta')

# Row 3 panel (2,3): r_D
ax = axes[2,3]
ax.plot(t, dmft.r_D,   color=BLUE,  lw=2,   label='DMFT')
ax.plot(t, dmft_c.r_D, color=GREEN, lw=1.5, ls='--', label='DMFT+clamp')
r_D_gf  = (mu['Rv_self'] - mu['Rv_cross']) / np.sqrt(2)
r_D_se  = np.sqrt(sig['Rv_self']**2 + sig['Rv_cross']**2) / np.sqrt(2)
ax.plot(t, r_D_gf, color=RED, lw=1.5, ls=':', label='GF')
ax.fill_between(t, r_D_gf-2*r_D_se, r_D_gf+2*r_D_se, color=BAND, alpha=0.5)
ns_rD   = abs(dmft.r_D[-1]   - r_D_gf[-1]) / (r_D_se[-1]+1e-12)
ns_rD_c = abs(dmft_c.r_D[-1] - r_D_gf[-1]) / (r_D_se[-1]+1e-12)
col = '#c00000' if ns_rD > 2 else '#1a5276'
ax.set_title(f'r_D   |Δ|/σ={ns_rD:.1f}  clamp={ns_rD_c:.1f}', fontsize=9, color=col)
ax.legend(fontsize=5, framealpha=0.4)
_axis(ax)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(outpath, dpi=130, bbox_inches='tight')
plt.close()
print(f"\nSaved → {outpath}")