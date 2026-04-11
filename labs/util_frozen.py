from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def render_value_and_policy_grid(V, pi, desc, title="Values + Policy", show_numbers=True, arrow_map=None):
    """
    Compact overlay: values plus policy arrows in each cell.
    """
    n = len(desc)
    assert V.size == n*n and pi.size == n*n
    if arrow_map is None:
        arrow_map = {0:'←', 1:'↓', 2:'→', 3:'↑'}

    V_grid = V.reshape((n, n))

    fig, ax = plt.subplots(figsize=(n*0.6, n*0.6))
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_aspect('equal'); ax.invert_yaxis()

    for i in range(n):
        for j in range(n):
            cell = desc[i][j]
            face = "black" if cell == "H" else "white"
            ax.add_patch(patches.Rectangle((j, i), 1, 1, linewidth=0.8, edgecolor="gray", facecolor=face))

            # Value text (small, top-ish)
            if show_numbers:
                color = "white" if cell == "H" else "black"
                ax.text(j+0.5, i+0.35, f"{V_grid[i,j]:.2f}", ha="center", va="center", color=color, fontsize=6.5)

            # Arrow (slightly above center)
            a = int(pi[i*n + j])
            arrow = arrow_map.get(a, '?')
            arrow_color = "white" if cell == "H" else "black"
            ax.text(j+0.5, i+0.62, arrow, ha="center", va="center", fontsize=9, color=arrow_color)

            # S/G marks
            if cell == "S":
                ax.text(j+0.5, i+0.18, "S", color="blue", ha="center", va="center", fontsize=8, fontweight='bold')
            elif cell == "G":
                ax.text(j+0.5, i+0.18, "G", color="red",  ha="center", va="center", fontsize=8, fontweight='bold')

    ax.set_xticks([]); ax.set_yticks([])
    if title: plt.title(title, fontsize=9, pad=2)
    plt.tight_layout(pad=0); plt.show()


def render_value_grid(V, desc, title=None, show_numbers=True):
    """
    Compact FrozenLake value grid.
    Holes (H) are black with white values, others are white with black values.
    """
    n = len(desc)
    assert V.size == n * n, f"Expected V length {n*n}, got {V.size}"
    V_grid = V.reshape((n, n))

    fig, ax = plt.subplots(figsize=(n * 0.6, n * 0.6))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    for i in range(n):
        for j in range(n):
            cell = desc[i][j]
            facecolor = "black" if cell == "H" else "white"
            rect = patches.Rectangle((j, i), 1, 1, linewidth=0.8,
                                     edgecolor="gray", facecolor=facecolor)
            ax.add_patch(rect)

            if show_numbers:
                color = "white" if cell == "H" else "black"
                ax.text(j + 0.5, i + 0.55, f"{V_grid[i, j]:.2f}",
                        ha="center", va="center", color=color, fontsize=7)

            if cell == "S":
                ax.text(j + 0.5, i + 0.25, "S", color="blue",
                        ha="center", va="center", fontsize=8, fontweight='bold')
            elif cell == "G":
                ax.text(j + 0.5, i + 0.25, "G", color="red",
                        ha="center", va="center", fontsize=8, fontweight='bold')

    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        plt.title(title, fontsize=9, pad=2)
    plt.tight_layout(pad=0)
    plt.show()


LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
ACTIONS = [LEFT, DOWN, RIGHT, UP]
DIRS = {
    LEFT:  (0, -1),
    DOWN:  (1, 0),
    RIGHT: (0, 1),
    UP:    (-1, 0),
}

def _grid_to_idx(r: int, c: int, ncols: int) -> int:
    return r * ncols + c

def _idx_to_grid(s: int, ncols: int) -> Tuple[int, int]:
    return divmod(s, ncols)

def _clip_move(r: int, c: int, dr: int, dc: int, nrows: int, ncols: int) -> Tuple[int, int]:
    rr, cc = r + dr, c + dc
    rr = min(max(rr, 0), nrows - 1)
    cc = min(max(cc, 0), ncols - 1)
    return rr, cc


# General transition builder for FrozenLake-style maps (works for any rectangular map).
# Compatible with Gymnasium's action order: LEFT=0, DOWN=1, RIGHT=2, UP=3
def build_frozenlake_transitions(desc: List[str], is_slippery: bool = True):
    """
    Build transition probabilities for a FrozenLake-like grid.

    Args:
        desc: list of strings (rows), made of {'S','F','H','G'}.
        is_slippery: if True -> stochastic: {left, forward, right} each with prob 1/3.
                     if False -> deterministic in the intended direction.

    Returns:
        P: np.ndarray (S, A, S), transition probabilities
        R: np.ndarray (S, A, S), rewards (1 on entering 'G', else 0)
        absorbing: np.ndarray (S,), True for 'H' or 'G' (absorbing/self-loop)
        shape_2d: (nrows, ncols)
        flatten_map: np.ndarray (S,), identity map (r,c)->s ordering
    """
    nrows = len(desc)
    ncols = len(desc[0])
    S = nrows * ncols
    A = 4

    grid = np.array([list(row) for row in desc])
    is_hole = (grid == 'H')
    is_goal = (grid == 'G')
    absorbing = (is_hole | is_goal).reshape(-1)

    P = np.zeros((S, A, S), dtype=float)
    R = np.zeros((S, A, S), dtype=float)

    def step_from_state(s: int, a: int) -> int:
        r, c = _idx_to_grid(s, ncols)
        dr, dc = DIRS[a]
        rr, cc = _clip_move(r, c, dr, dc, nrows, ncols)
        return _grid_to_idx(rr, cc, ncols)

    for s in range(S):
        if absorbing[s]:
            # Absorbing states self-loop for all actions
            for a in ACTIONS:
                P[s, a, s] = 1.0
            continue

        for a in ACTIONS:
            if is_slippery:
                left = (a - 1) % 4
                right = (a + 1) % 4
                for aa in [left, a, right]:
                    s2 = step_from_state(s, aa)
                    P[s, a, s2] += 1.0/3.0
            else:
                s2 = step_from_state(s, a)
                P[s, a, s2] = 1.0

            # Reward for ARRIVING at goal
            for s2 in range(S):
                if P[s, a, s2] > 0:
                    rr, cc = _idx_to_grid(s2, ncols)
                    if grid[rr, cc] == 'G':
                        R[s, a, s2] = 1.0

    flatten_map = np.arange(S, dtype=int)
    return P, R, absorbing, (nrows, ncols), flatten_map