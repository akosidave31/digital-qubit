"""A circuit: an editable list of gates that runs on the N-qubit engines.
Pure NumPy (no UI) so it can be tested; the app is built on top of it."""
from .nqubit import NQubit, NDensity, gate_matrix

TWO = ("cx", "cz", "swap")
LABEL = {"h": "H", "x": "X", "y": "Y", "z": "Z", "s": "S", "t": "T",
         "rx": "Rx", "ry": "Ry", "rz": "Rz", "p": "P"}


class Circuit:

    def __init__(self, n):
        n = int(n)
        if not 1 <= n <= 26:
            raise ValueError("number of qubits must be 1..26")
        self.n = n
        self.ops = []

    def _q(self, q):
        if not (isinstance(q, int) and 0 <= q < self.n):
            raise ValueError(f"qubit {q} does not exist (0..{self.n - 1})")

    def add(self, name, q, param=None):
        """Single-qubit gate: h x y z s t, or rx ry rz p with an angle."""
        name = name.lower()
        if name in TWO:
            raise ValueError(f"'{name}' is a two-qubit gate: use add2")
        gate_matrix(name, param)
        q = int(q)
        self._q(q)
        self.ops.append((name, q, None if param is None else float(param)))
        return self

    def add2(self, name, a, b):
        """Two-qubit gate: cx (a = control, b = target), cz, swap."""
        name = name.lower()
        if name not in TWO:
            raise ValueError(f"'{name}' is not a two-qubit gate")
        a, b = int(a), int(b)
        self._q(a)
        self._q(b)
        if a == b:
            raise ValueError("a two-qubit gate needs two different qubits")
        self.ops.append((name, a, b))
        return self

    def undo(self):
        if self.ops:
            self.ops.pop()
        return self

    def clear(self):
        self.ops = []
        return self

    def _run(self, reg):
        for name, a, b in self.ops:
            reg = getattr(reg, name)(a, b) if name in TWO else reg.gate(name, a, b)
        return reg

    def state(self):
        """Final pure state (no noise)."""
        return self._run(NQubit(self.n))

    def noisy(self, t, T1, T_phi=None):
        """Run the gates, then every qubit waits time t under T1/T_phi noise."""
        return NDensity.from_state(self.state()).idle(t, T1, T_phi)

    def to_text(self):
        """Text diagram, one line per qubit."""
        cols = []
        for name, a, b in self.ops:
            col = ["" for _ in range(self.n)]
            if name in TWO:
                m = {"cx": ("●", "⊕"), "cz": ("●", "●"), "swap": ("×", "×")}[name]
                col[a], col[b] = m
                for q in range(min(a, b) + 1, max(a, b)):
                    col[q] = "┼"
            else:
                col[a] = LABEL[name] + ("" if b is None else f"({b:.2f})")
            cols.append(col)
        lines = [f"q{q}: " for q in range(self.n)]
        for col in cols:
            w = max(len(c) for c in col)
            for q in range(self.n):
                lines[q] += "─" + (col[q] or "─").center(w, "─") + "─"
        return "\n".join(lines)
