from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
src = here / "results_assignment3"
s = pd.read_csv(src / "summary.csv")
r = pd.read_csv(src / "raw_times.csv")
folder = here / "figures"
folder.mkdir(exist_ok=True)
sizes = sorted(s.n.unique())
threads = [1, 2, 4, 8]
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .2, "axes.spines.top": False, "axes.spines.right": False})

def finish(fig, ax, name):
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(folder / name, dpi=200)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 3.7))
for name, p in [("Sequential", 0), ("1 threads", 1), ("2 threads", 2), ("4 threads", 4), ("8 threads", 8)]:
    x = s[s.threads == p].sort_values("n")
    ax.plot(x.n, x.median_ms, "o-", label=name)
ax.set_yscale("log"); ax.set_xticks(sizes); ax.set_xlabel("Matrix side length n"); ax.set_ylabel("Median time (ms, log scale)")
ax.set_title("Execution time by matrix size and thread count")
finish(fig, ax, "fig1_runtime.png")

fig, ax = plt.subplots(figsize=(7.2, 3.7))
for n in sizes:
    x = s[(s.n == n) & (s.threads > 0)].sort_values("threads")
    ax.plot(x.threads, x.speedup, "o-", label=f"n={n}")
ax.plot(threads, threads, "--", color="0.45", label="Ideal linear")
ax.set_xticks(threads); ax.set_xlabel("Parallel worker threads"); ax.set_ylabel("Speedup (Tseq / Tpar)")
ax.set_title("Speedup relative to equivalent sequential kernel")
finish(fig, ax, "fig2_speedup.png")

fig, ax = plt.subplots(figsize=(7.2, 3.7))
for n in sizes:
    x = s[(s.n == n) & (s.threads > 0)].sort_values("threads")
    ax.plot(x.threads, x.efficiency, "o-", label=f"n={n}")
ax.axhline(1, linestyle="--", color="0.45", label="100% ideal")
ax.set_xticks(threads); ax.set_xlabel("Parallel worker threads"); ax.set_ylabel("Efficiency (speedup / threads)")
ax.set_title("Parallel efficiency and anomalous >1 values")
finish(fig, ax, "fig3_efficiency.png")

fig, ax = plt.subplots(figsize=(7.2, 3.7))
seq = s[s.threads == 0].sort_values("n")
p8 = s[s.threads == 8].sort_values("n")
ax.plot(seq.n, seq.median_heap_delta_mib, "o-", label="Sequential: median retained heap delta")
ax.plot(p8.n, p8.median_heap_delta_mib, "s--", label="8 threads: median retained heap delta")
ax.plot(seq.n, seq.estimated_three_matrices_mib, "^:", label="Estimated A+B+C numeric payload")
ax.set_xticks(sizes); ax.set_xlabel("Matrix side length n"); ax.set_ylabel("MiB")
ax.set_title("Two different memory quantities (not equivalent)")
finish(fig, ax, "fig4_memory.png")

fig, ax = plt.subplots(figsize=(7.2, 3.45))
groups = [r[(r.n == 512) & (r.threads == x)].time_ms.to_numpy() for x in [0,1,2,4,8]]
ax.boxplot(groups, tick_labels=["Sequential", "1 thread", "2 threads", "4 threads", "8 threads"])
ax.set_ylabel("Single-run time (ms)")
ax.set_title("Variability in individual runs, n=512 (5 runs each)")
ax.grid(False)
fig.tight_layout(); fig.savefig(folder / "fig5_variability_512.png", dpi=200); plt.close(fig)
print("Created five figures in:", folder)
