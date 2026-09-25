"""Interactive circuit app for Jupyter / Colab.
Needs ipywidgets and matplotlib (both preinstalled in Colab).
Usage:  from digital_qubit.app import launch; launch()"""
import numpy as np
from .circuit import Circuit, TWO

ANGLE_GATES = ("rx", "ry", "rz", "p")


def _bloch(ax, r, title):
    u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                      color="lightgray", linewidth=0.4)
    for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        ax.plot([-d[0], d[0]], [-d[1], d[1]], [-d[2], d[2]], color="gray", linewidth=0.6)
    if np.linalg.norm(r) > 1e-6:
        ax.quiver(0, 0, 0, r[0], r[1], r[2], color="tab:purple", linewidth=2, arrow_length_ratio=0.15)
    else:
        ax.scatter([0], [0], [0], color="tab:purple", s=30)
    ax.text(0, 0, 1.25, "|0>", ha="center", fontsize=8)
    ax.text(0, 0, -1.4, "|1>", ha="center", fontsize=8)
    ax.text(1.3, 0, 0, "x", fontsize=7)
    ax.text(0, 1.3, 0, "y", fontsize=7)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.axis("off")
    ax.set_title(title, fontsize=9)


def launch(max_qubits=5):
    import ipywidgets as w
    import matplotlib.pyplot as plt
    from IPython.display import display

    rng = np.random.default_rng()
    st = {"c": Circuit(2), "counts": None}
    L = lambda: w.Layout(width="95%")
    SD = {"description_width": "70px"}

    n_w = w.IntSlider(value=2, min=1, max=max_qubits, description="qubits", layout=L(), style=SD)
    gate_w = w.Dropdown(options=["h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "p", "cx", "cz", "swap"],
                        value="h", description="gate", layout=L(), style=SD)
    q_w = w.Dropdown(options=[0, 1], value=0, description="qubit", layout=L(), style=SD)
    q2_w = w.Dropdown(options=[0, 1], value=1, description="target", layout=L(), style=SD)
    ang_w = w.FloatSlider(value=round(np.pi / 2, 2), min=0, max=round(2 * np.pi, 2), step=0.01,
                          description="angle", layout=L(), style=SD)
    add_b = w.Button(description="Add gate", button_style="primary")
    undo_b, clear_b = w.Button(description="Undo"), w.Button(description="Clear")
    meas_b = w.Button(description="Measure 1000 shots", layout=L())
    noise_w = w.Checkbox(value=False, description="noise: wait before measuring", indent=False, layout=L())
    t_w = w.FloatSlider(value=10, min=0, max=150, step=1, description="wait t", layout=L(), style=SD)
    T1_w = w.BoundedFloatText(value=50, min=0.1, max=1e6, description="T1", layout=L(), style=SD)
    Tp_w = w.BoundedFloatText(value=40, min=0.1, max=1e6, description="T_phi", layout=L(), style=SD)
    msg = w.HTML()
    out = w.Output(layout=w.Layout(width="100%", overflow="hidden"))

    def register():
        c = st["c"]
        return c.noisy(t_w.value, T1_w.value, Tp_w.value) if noise_w.value else c.state()

    def redraw(*_):
        with out:
            out.clear_output(wait=True)
            c = st["c"]
            reg = register()
            n = c.n
            print(c.to_text() if c.ops else "empty circuit:\nall qubits in |0>")
            if noise_w.value:
                print(f"noise on: wait t = {t_w.value:g}\n(T1 = {T1_w.value:g}, T_phi = {Tp_w.value:g})")
            cols = min(n, 3)
            rows = -(-n // cols)
            fig = plt.figure(figsize=(6.0, 2.3 * rows))
            for q in range(n):
                ax = fig.add_subplot(rows, cols, q + 1, projection="3d")
                r = reg.bloch(q)
                _bloch(ax, r, f"q{q}: length {np.linalg.norm(r):.2f}")
            plt.tight_layout()
            plt.show()
            p = reg.probs()
            labels = [reg.bits(i) for i in range(2 ** n)]
            fig, ax = plt.subplots(figsize=(6.0, 2.8))
            ax.bar(labels, p, color="tab:purple", label="probability")
            if st["counts"] is not None:
                ax.plot(labels, st["counts"], "o", color="tab:orange", label="1000 shots")
                ax.legend(fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_ylabel("P")
            plt.xticks(rotation=90 if n > 3 else 0, fontsize=7 if n > 3 else 9)
            plt.tight_layout()
            plt.show()
            print("length < 1: entangled\nor affected by noise")

    def refresh_controls(*_):
        g = gate_w.value
        two = g in TWO
        q2_w.layout.display = "" if two else "none"
        ang_w.layout.display = "" if g in ANGLE_GATES else "none"
        q_w.description = {"cx": "control", "cz": "qubit A", "swap": "qubit A"}.get(g, "qubit")
        q2_w.description = "target" if g == "cx" else "qubit B"

    def changed():
        st["counts"] = None
        redraw()

    def on_add(_):
        g = gate_w.value
        try:
            if g in TWO:
                st["c"].add2(g, q_w.value, q2_w.value)
            elif g in ANGLE_GATES:
                st["c"].add(g, q_w.value, ang_w.value)
            else:
                st["c"].add(g, q_w.value)
            msg.value = ""
        except ValueError as e:
            msg.value = f"<span style='color:#b00'>{e}</span>"
        changed()

    def on_n(change):
        n = change["new"]
        st["c"] = Circuit(n)
        q_w.options = list(range(n))
        q2_w.options = list(range(n))
        q_w.value = 0
        q2_w.value = 1 if n > 1 else 0
        msg.value = ""
        changed()

    def on_measure(_):
        reg = register()
        st["counts"] = np.bincount(reg.sample(1000, rng), minlength=2 ** reg.n) / 1000
        redraw()

    add_b.on_click(on_add)
    undo_b.on_click(lambda _: (st["c"].undo(), changed()))
    clear_b.on_click(lambda _: (st["c"].clear(), changed()))
    meas_b.on_click(on_measure)
    n_w.observe(on_n, names="value")
    gate_w.observe(refresh_controls, names="value")
    for wd in (noise_w, t_w, T1_w, Tp_w):
        wd.observe(lambda _: changed(), names="value")

    refresh_controls()
    display(w.VBox([n_w, gate_w, q_w, q2_w, ang_w,
                    w.HBox([add_b, undo_b, clear_b]), meas_b,
                    noise_w, t_w, T1_w, Tp_w, msg, out], layout=w.Layout(width="100%")))
    redraw()


def launch_checker(model, device, tol=0.03):
    """Feed experiments to the neural model and the true physics; check the model's output."""
    import ipywidgets as w
    import matplotlib.pyplot as plt
    from IPython.display import display
    from .checker import Checker, BASIS_NAME

    ck = Checker(model, device, tol)
    rng = np.random.default_rng()
    L = lambda: w.Layout(width="95%")
    SD = {"description_width": "70px"}
    a_w = w.FloatSlider(value=1.57, min=0, max=3.14, step=0.01, description="tilt a", layout=L(), style=SD)
    b_w = w.FloatSlider(value=0.0, min=0, max=6.28, step=0.01, description="phase b", layout=L(), style=SD)
    t_w = w.FloatSlider(value=20, min=0, max=ck.t_max, step=1, description="wait t", layout=L(), style=SD)
    k_w = w.Dropdown(options=[("X", 0), ("Y", 1), ("Z", 2)], value=0, description="measure", layout=L(), style=SD)
    run_b = w.Button(description="Run this experiment", button_style="primary", layout=L())
    batch_b = w.Button(description="Run 500 random experiments", layout=L())
    out = w.Output(layout=w.Layout(width="100%", overflow="hidden"))

    def on_run(_):
        with out:
            out.clear_output(wait=True)
            a, b, t, k = a_w.value, b_w.value, t_w.value, k_w.value
            r = ck.run(a, b, t, k, 1000, rng)
            print(f"experiment: a={a:.2f} b={b:.2f}\n  wait t={t:g}, measure {BASIS_NAME[k]}")
            print(f"neural model: P(1) = {r['model']:.3f} +/- {r['sd']:.3f}")
            print(f"true physics: P(1) = {r['truth']:.3f}")
            print(f"measured:     {r['measured']:.3f} (1000 shots, +/-{r['shot_sd']:.3f})")
            print(f"model error:  {r['error']:+.3f}")
            print("MATCH: model output agrees with physics" if r["agree"]
                  else f"MISMATCH: error above {tol}")
            s = ck.sweep(a, b, k, np.linspace(0, ck.t_max, 61), 1000, rng)
            fig, ax = plt.subplots(figsize=(6.0, 3.2))
            ax.fill_between(s["t"], s["model"] - 2 * s["sd"], s["model"] + 2 * s["sd"],
                            color="tab:purple", alpha=0.25, label="model +/- 2 sd")
            ax.plot(s["t"], s["truth"], "k-", lw=1.5, label="true physics")
            ax.plot(s["t"], s["model"], "--", color="tab:purple", lw=1.5, label="neural model")
            ax.plot(s["t"][::4], s["measured"][::4], "o", color="tab:orange", ms=4, label="measured")
            ax.plot([t], [r["model"]], "*", color="tab:red", ms=12, label="this experiment")
            ax.set_xlabel("wait time t")
            ax.set_ylabel("P(1)")
            ax.set_ylim(-0.02, 1.02)
            ax.legend(fontsize=7, loc="best")
            plt.tight_layout()
            plt.show()

    def on_batch(_):
        with out:
            out.clear_output(wait=True)
            print("running 500 random experiments...")
            res = ck.batch(500, 1000, rng)
            S = res["summary"]
            out.clear_output(wait=True)
            print(f"500 random experiments -> {S['verdict']}")
            print(f"agree within {tol}: {S['agree_fraction'] * 100:.1f}%")
            print(f"mean error {S['mean_abs_error']:.4f}, worst {S['max_abs_error']:.4f}")
            if S["coverage_2sd"] is not None:
                print(f"truth inside model +/- 2 sd: {S['coverage_2sd'] * 100:.1f}%")
            better = S["rmse"] < S["measured_rmse"]
            print(f"model RMSE {S['rmse']:.4f} vs 1000-shot\n  measurement {S['measured_rmse']:.4f}"
                  f" -> model is {'MORE' if better else 'LESS'} accurate")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 3.0))
            ax1.plot([0, 1], [0, 1], "k-", lw=0.8)
            ax1.scatter(res["truth"], res["model"], s=6, color="tab:purple", alpha=0.6)
            ax1.set_xlabel("true P(1)")
            ax1.set_ylabel("model P(1)")
            ax1.set_title("model vs physics", fontsize=9)
            ax2.hist(res["error"], bins=30, color="tab:purple")
            ax2.axvline(-tol, color="tab:red", lw=0.8)
            ax2.axvline(tol, color="tab:red", lw=0.8)
            ax2.set_xlabel("model error")
            ax2.set_title("error spread", fontsize=9)
            plt.tight_layout()
            plt.show()

    run_b.on_click(on_run)
    batch_b.on_click(on_batch)
    display(w.VBox([a_w, b_w, t_w, k_w, run_b, batch_b, out], layout=w.Layout(width="100%")))
    on_run(None)
