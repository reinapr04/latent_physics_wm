"""Generate docs/architecture.svg, the system in one picture (render to png with a browser or rsvg-convert)."""
import html

W, H = 1800, 1120
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURF, BOX, BOXS = "#fcfcfb", "#f3f2ee", "#d9d7d1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_T, ORANGE_T, AQUA_T = "#e6effb", "#fdece5", "#e3f6ee"
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
out = []


def esc(s):
    return html.escape(s, quote=False)


def text(x, y, s, size=15, color=INK, weight="normal", anchor="start", italic=False):
    style = f"font-family:{FONT};font-size:{size}px;fill:{color};font-weight:{weight};" + ("font-style:italic;" if italic else "")
    out.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="{style}">{esc(s)}</text>')


def box(x, y, w, h, title, lines, fill=BOX, stroke=BOXS, title_color=INK, size=14.5, lh=20, accent=None):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" ry="10" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    if accent:
        out.append(f'<rect x="{x}" y="{y}" width="6" height="{h}" rx="3" fill="{accent}"/>')
    text(x + 16, y + 26, title, size=16, color=title_color, weight="bold")
    for i, l in enumerate(lines):
        text(x + 16, y + 52 + i * lh, l, size=size, color=INK2)


def arrow(x1, y1, x2, y2, color=INK2, label=None, lx=None, ly=None, width=2, dashed=False, size=13.5):
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" marker-end="url(#ah-{color[1:]})"{dash}/>')
    if label:
        text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2 - 8, label, size=size, color=INK2, anchor="middle")


def poly(points, color=INK2, width=2, label=None, lx=0, ly=0, dashed=False, size=13.5):
    pts = " ".join(f"{x},{y}" for x, y in points)
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" marker-end="url(#ah-{color[1:]})"{dash}/>')
    if label:
        text(lx, ly, label, size=size, color=INK2, anchor="middle")


def pill(x, y, label, size=13.5, color=INK2):
    w = 7.4 * len(label) * size / 13.5 + 16
    out.append(f'<rect x="{x - w / 2}" y="{y - size}" width="{w}" height="{size + 8}" rx="5" fill="{SURF}"/>')
    text(x, y, label, size=size, color=color, anchor="middle")


def band(x, y, w, h, label, color):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="none" stroke="{color}" stroke-width="1.5" stroke-dasharray="3 6" opacity="0.9"/>')
    out.append(f'<rect x="{x + 14}" y="{y - 13}" width="{9 * len(label) + 20}" height="26" rx="6" fill="{SURF}"/>')
    text(x + 24, y + 5, label, size=16, color=color, weight="bold")


# ------------------------------------------------------------------ header
out.append(f'<rect width="{W}" height="{H}" fill="{SURF}"/>')
defs = "".join(
    f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
    f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{c}"/></marker>' for c in [INK2, BLUE, ORANGE, AQUA])
out.append(f"<defs>{defs}</defs>")
text(40, 46, "Latent physics world model for pushing", size=30, weight="bold")
text(40, 74, "A push-dynamics model conditioned on a learned per-environment physics latent, planned with MPC, "
             "and adapted to unseen physics from a few probing pushes.", size=16, color=INK2)

# ------------------------------------------------------------------ row 1: environment
Y1 = 118
band(30, Y1, 1740, 252, "1  Environment and data  (ManiSkill 3, GPU-parallel, 32 envs)", INK2)
box(60, Y1 + 30, 380, 180, "Scene with hidden physics",
    ["Panda arm, table, one object per env:",
     "4 cm cube (block task) or 3.5 cm ball (roll task)",
     "mass, static/dynamic friction and, for the ball,",
     "rolling resistance drawn per env; never given",
     "to the model. 64–128 settings in a pool."], accent=INK2)
box(500, Y1 + 30, 330, 180, "Observation",
    ["fixed camera, 256 × 256",
     "→ world-frame point cloud",
     "   (object + table, by height, no segmentation)",
     "→ RGB image (image-model ablations)"], accent=BLUE)
box(890, Y1 + 30, 420, 180, "Push primitive  =  one action",
    ["a = (p_contact ∈ ℝ³,  d_push ∈ ℝ²,  F [N],  T [s])",
     "lift → travel → descend → approach (until contact)",
     "→ push with force control for T → back off",
     "→ home (→ settle until the ball stops)",
     "knows nothing about the object; 7–9 s simulated"], accent=INK2)
box(1370, Y1 + 30, 370, 180, "Transition  (s, a, s′)",
    ["cloud before the push, the push,",
     "the object's rigid motion (R, t)",
     "— all in the push's action frame —",
     "appended to the dataset D",
     "(~3.7k per run; 8.6k in the big run)"], accent=INK2)
arrow(440, Y1 + 120, 500, Y1 + 120, label="render")
arrow(1310, Y1 + 120, 1370, Y1 + 120, label="outcome")
poly([(1100, Y1 + 210), (1100, Y1 + 230), (250, Y1 + 230), (250, Y1 + 212)], color=INK2)
pill(675, Y1 + 235, "the planner's push is executed in the scene")

# ------------------------------------------------------------------ row 2: model
Y2 = 408
band(30, Y2, 1740, 330, "2  Dynamics model f_θ  — trained on D with a point-flow loss", BLUE)
cy = Y2 + 110
box(60, Y2 + 34, 250, 150, "Action frame",
    ["origin = contact point on the table,",
     "x = push direction, z up.",
     "The cloud is expressed here, so the",
     "push itself reduces to (p_z, F, T)."], fill=BLUE_T, stroke="#bcd3f3", accent=BLUE)
box(360, Y2 + 34, 230, 150, "Input cloud · 512 pts",
    ["all points above the table",
     "(the object) + 100 table points,",
     "farthest-point sampled;",
     "mirror + noise augmentation"], fill=BLUE_T, stroke="#bcd3f3", accent=BLUE)
box(640, Y2 + 34, 230, 150, "Encoder → 128-d",
    ["PointNet (per-point MLP + max)",
     "variants: early fusion, cross-",
     "attention, transformer, wide /",
     "residual heads"], fill=BLUE_T, stroke="#bcd3f3", accent=BLUE)
box(920, Y2 + 34, 250, 150, "Head MLP 256 × 3",
    ["input [feature, c]",
     "c = [(p_z, F, T) / scale,  z]",
     "last layer zero-initialised",
     "(starts by predicting no motion)"], fill=BOX, accent=INK2)
box(1220, Y2 + 34, 230, 150, "Output (R, t)",
    ["rigid motion of the object",
     "in the action frame",
     "(axis-angle + translation;",
     "ball: translation only)"], fill=BOX, accent=INK2)
box(1500, Y2 + 34, 240, 150, "Point-flow loss",
    ["move the object's observed",
     "points by predicted and by true",
     "(R, t): mean distance in metres.",
     "AdamW, early stopping on val"], fill=BOX, accent=INK2)
for x1, x2 in [(310, 360), (590, 640), (870, 920), (1170, 1220), (1450, 1500)]:
    arrow(x1, cy, x2, cy, color=BLUE)
# image alternative (under the cloud boxes)
box(60, Y2 + 214, 560, 96, "Alternative inputs (ablations)",
    ["frozen DINOv2 tokens or a small CNN on the RGB image, in the world frame",
     "or on an action-frame crop (rectified to the table plane, centred on the push)"],
    fill=BOX, stroke=BOXS, accent=MUTED)
arrow(475, Y2 + 214, 475, Y2 + 186, color=MUTED, dashed=True)
text(488, Y2 + 204, "replaces the cloud", size=13, color=MUTED)
# latent table
box(640, Y2 + 214, 530, 96, "Physics latent  z ∈ ℝ⁸  —  one embedding row per physics setting",
    ["learned only by backprop through the dynamics loss (small L2 prior);",
     "the mean of the training latents is the “no adaptation” latent."],
    fill=ORANGE_T, stroke="#f5c9b6", title_color="#b4431b", accent=ORANGE)
arrow(1045, Y2 + 214, 1045, Y2 + 186, color=ORANGE)
text(1056, Y2 + 204, "z", size=15, color="#b4431b", weight="bold")
# ensemble
box(1220, Y2 + 214, 520, 96, "Ensemble (optional)",
    ["5 independently trained models (own seeds, own latents); the planner and",
     "the latent fit use the mean of their predictions — the best model in control."],
    fill=BOX, stroke=BOXS, accent=INK2)

# ------------------------------------------------------------------ row 3: use
Y3 = 782
band(30, Y3, 1740, 300, "3  Using the model", AQUA)
box(60, Y3 + 30, 560, 230, "Planning: random-shooting MPC",
    ["every push:  sample 1024 candidate pushes on the observed cloud",
     "(aimed at the object's outline + some anywhere)",
     "→ f_θ imagines each outcome: move the object's points by (R, t)",
     "→ score the imagined object position with the task reward",
     "→ execute the best push with the primitive → observe → replan",
     "horizon 1; uses only the observation, the model and z."],
    fill=AQUA_T, stroke="#b6e6d2", title_color="#0e7a53", accent=AQUA)
box(680, Y3 + 30, 500, 230, "Test-time adaptation to unseen physics",
    ["new environment, physics never seen in training",
     "→ k probing pushes through the object's centre",
     "   (forces log-spread around where it starts to move)",
     "→ fit z only, model frozen (Adam, 200 steps, same loss)",
     "→ plan with the fitted z.",
     "Compared against the mean latent and an oracle fit."],
    fill=ORANGE_T, stroke="#f5c9b6", title_color="#b4431b", accent=ORANGE)
box(1240, Y3 + 30, 500, 230, "Training loop (model-based RL, on-policy data)",
    ["1. 4 rollouts of random pushes → D",
     "2. repeat 10×:  train f_θ + latents on D (early stopping)",
     "               2 rollouts of MPC pushing to goals → D",
     "3. final training phase",
     "settings with the least data are assigned to the 32 envs",
     "each rollout, so every latent keeps being trained."],
    fill=BOX, accent=INK2)
arrow(680, Y3 + 110, 620, Y3 + 110, color=ORANGE)
text(650, Y3 + 98, "fitted z", size=13.5, color="#b4431b", anchor="middle", weight="bold")
poly([(1240, Y3 + 200), (1210, Y3 + 200), (1210, Y3 + 274), (340, Y3 + 274), (340, Y3 + 262)], color=AQUA)
pill(925, Y3 + 279, "MPC rollouts collect the data the model is trained on", color="#0e7a53")
# arrows between rows
arrow(1555, Y1 + 210, 1555, Y2 - 6)
pill(1605, (Y1 + 210 + Y2) / 2 + 8, "dataset D", size=14)
arrow(340, Y2 + 318, 340, Y3 - 6, color=AQUA)
text(356, Y2 + 352, "f_θ(cloud, push, z) → predicted motion of the object", size=14, color=INK2)

text(40, H - 16, "Repo: env/ (scene, primitive) · dynamics_model/ (frame, encoders, latent, loss) · policy/mpc.py · "
                 "train_dynamics.py · eval_latent_inference.py · eval_mpc.py", size=13, color=MUTED)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">' + "".join(out) + "</svg>")
open(__import__("pathlib").Path(__file__).resolve().parents[1] / "docs" / "architecture.svg", "w").write(svg)
print("svg written", len(svg))
