import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ======================
# Parameters
# ======================
WORLD_SIZE = 100
N_BLOBS = 12
N_FOOD = 60                     # initial food
# Movement / energy economics
ENERGY_LOSS_BASE = 0.6
SPEED_COST_COEFF = 0.15
REPRODUCTION_THRESHOLD = 120
INITIAL_ENERGY = 80
FOOD_ENERGY = 50
EAT_RADIUS = 2.2
MUTATION_STD_SPEED = 0.15
MUTATION_STD_VISION = 0.8
MUTATE_STRATEGY_P = 0.07

# Food ecology controls (limited respawn)
MAX_FOOD = 80            # carrying capacity (hard cap)
FOOD_SPAWN_PER_TICK = 2  # max spawn attempts per frame
FOOD_SPAWN_PROB = 0.5    # success probability per attempt

MAX_FRAMES = 1500
RNG = np.random.default_rng()

# Strategies and a FIXED color mapping (shared by scatter + lines + legend)
STRAT_ORDER = ["cardinal", "diagonal", "knight", "random"]
STRAT_TO_ID = {s:i for i,s in enumerate(STRAT_ORDER)}
cmap = plt.get_cmap("tab10")
STRAT_COLOR = {s: cmap(STRAT_TO_ID[s] % cmap.N) for s in STRAT_ORDER}

# ======================
# Helpers
# ======================
def random_position():
    return RNG.uniform(0, WORLD_SIZE, size=2)

def unit_vec(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros_like(v)

CARDINAL_DIRS = np.array([[1,0],[-1,0],[0,1],[0,-1]])
DIAGONAL_DIRS  = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
KNIGHT_DIRS = np.array([
    [ 2, 1],[ 2,-1],[-2, 1],[-2,-1],
    [ 1, 2],[ 1,-2],[-1, 2],[-1,-2]
], dtype=float)

def pick_strategy_step(strategy, last_dir=None):
    if strategy == "cardinal":
        dirs = CARDINAL_DIRS
        if last_dir is not None and RNG.random() < 0.7:
            return last_dir
        return dirs[RNG.integers(len(dirs))]
    if strategy == "diagonal":
        dirs = DIAGONAL_DIRS
        if last_dir is not None and RNG.random() < 0.7:
            return last_dir
        return dirs[RNG.integers(len(dirs))]
    if strategy == "knight":
        return KNIGHT_DIRS[RNG.integers(len(KNIGHT_DIRS))]
    # fallback random
    angle = RNG.uniform(0, 2*np.pi)
    return np.array([np.cos(angle), np.sin(angle)])

# ======================
# Blob
# ======================
class Blob:
    def __init__(self, speed, vision, strategy):
        self.pos = random_position()
        self.energy = INITIAL_ENERGY
        self.speed = max(0.1, float(speed))
        self.vision = max(0.0, float(vision))
        self.strategy = strategy
        self._last_dir = None

    def move(self, foods):
        # Chase nearest food if within vision; else follow strategy
        target_vec = None
        if foods:
            vx = self.vision
            x, y = self.pos
            candidates = []
            for f in foods:
                if abs(f[0]-x) <= vx and abs(f[1]-y) <= vx:
                    candidates.append(f)
            if candidates:
                dists = [np.linalg.norm(self.pos - f) for f in candidates]
                i = int(np.argmin(dists))
                if dists[i] <= self.vision:
                    target_vec = unit_vec(candidates[i] - self.pos)

        if target_vec is None:
            step_dir = pick_strategy_step(self.strategy, self._last_dir)
            step_dir = unit_vec(step_dir)
            self._last_dir = step_dir
        else:
            step_dir = target_vec
            self._last_dir = step_dir

        step = step_dir * self.speed
        self.pos = np.clip(self.pos + step, 0, WORLD_SIZE)

        knight_surcharge = 0.4 if self.strategy == "knight" else 0.0
        move_cost = ENERGY_LOSS_BASE + SPEED_COST_COEFF * self.speed + knight_surcharge
        self.energy -= move_cost

    def eat(self, foods):
        for i, food in enumerate(foods):
            if np.linalg.norm(self.pos - food) < EAT_RADIUS:
                self.energy += FOOD_ENERGY
                foods.pop(i)
                return True
        return False

    def reproduce(self):
        if self.energy > REPRODUCTION_THRESHOLD:
            self.energy *= 0.5
            new_speed = max(0.1, self.speed + RNG.normal(0, MUTATION_STD_SPEED))
            new_vision = max(0.0, self.vision + RNG.normal(0, MUTATION_STD_VISION))
            new_strategy = self.strategy
            if RNG.random() < MUTATE_STRATEGY_P:
                options = STRAT_ORDER.copy()
                options.remove(self.strategy)
                new_strategy = options[RNG.integers(len(options))]
            return Blob(new_speed, new_vision, new_strategy)
        return None

# ======================
# World state
# ======================
INITIAL_STRATEGIES = ["cardinal","diagonal"]  # seed with two; others can evolve
def random_strategy():
    return INITIAL_STRATEGIES[RNG.integers(len(INITIAL_STRATEGIES))]

blobs = [Blob(speed=RNG.uniform(0.6, 1.4),
              vision=RNG.uniform(6.0, 14.0),
              strategy=random_strategy())
         for _ in range(N_BLOBS)]

foods = [random_position() for _ in range(N_FOOD)]

# ======================
# Figure with two subplots (ONE figure fixes update issues)
# ======================
fig, axes = plt.subplots(
    2, 2, figsize=(13, 8),
    gridspec_kw={"width_ratios":[1.1,0.9], "height_ratios":[1.2,1.0]}
)
ax_world, ax_ratio = axes[0]
ax_speed, ax_vision = axes[1]

# --- World subplot ---
scat_blobs = ax_world.scatter([], [], s=30)  # we'll set colors manually
scat_food  = ax_world.scatter([], [], c='green', s=20, marker='x')
ax_world.set_xlim(0, WORLD_SIZE); ax_world.set_ylim(0, WORLD_SIZE)
ax_world.set_title("World: blobs & food")

# Legend with fixed colors
legend_handles = []
for s in STRAT_ORDER:
    legend_handles.append(plt.Line2D([0],[0], marker='o', linestyle='',
                                     markerfacecolor=STRAT_COLOR[s],
                                     markeredgecolor='none', label=s))
ax_world.legend(handles=legend_handles, title="Strategy", loc="upper right", frameon=True)

# --- Ratio subplot ---
ax_ratio.set_title("Population ratios by strategy")
ax_ratio.set_xlabel("Frame")
ax_ratio.set_ylabel("Ratio (0-1)")
ax_ratio.set_ylim(0, 1.0)
ax_ratio.set_xlim(0, 100)  # will extend dynamically
ax_ratio.set_autoscalex_on(False)

ratio_lines = {}
for s in STRAT_ORDER:
    (line,) = ax_ratio.plot([], [], label=s, color=STRAT_COLOR[s])  # exact same colors
    ratio_lines[s] = line
ax_ratio.legend(loc="upper right", frameon=True)

# --- Avg speed subplot ---
ax_speed.set_title("Average speed by strategy")
ax_speed.set_xlabel("Frame")
ax_speed.set_ylabel("Speed")
ax_speed.set_xlim(0, 100)
ax_speed.set_autoscalex_on(False)
avg_speed_lines = {}
for s in STRAT_ORDER:
    (line,) = ax_speed.plot([], [], label=s, color=STRAT_COLOR[s])
    avg_speed_lines[s] = line
ax_speed.legend(loc="upper right", frameon=True)

# --- Avg vision subplot ---
ax_vision.set_title("Average vision by strategy")
ax_vision.set_xlabel("Frame")
ax_vision.set_ylabel("Vision")
ax_vision.set_xlim(0, 100)
ax_vision.set_autoscalex_on(False)
avg_vision_lines = {}
for s in STRAT_ORDER:
    (line,) = ax_vision.plot([], [], label=s, color=STRAT_COLOR[s])
    avg_vision_lines[s] = line
ax_vision.legend(loc="upper right", frameon=True)

# History buffers
history_frames = []
history_ratios = {s: [] for s in STRAT_ORDER}
history_avg_speed = {s: [] for s in STRAT_ORDER}
history_avg_vision = {s: [] for s in STRAT_ORDER}
paused = False  # track pause state for the animation

# ======================
# Food respawn (limited)
# ======================
def limited_food_respawn():
    if len(foods) < MAX_FOOD:
        for _ in range(FOOD_SPAWN_PER_TICK):
            if len(foods) >= MAX_FOOD:
                break
            if RNG.random() < FOOD_SPAWN_PROB:
                foods.append(random_position())

def compute_and_record_stats(frame):
    total = max(1, len(blobs))
    counts = {s: 0 for s in STRAT_ORDER}
    speed_sums = {s: 0.0 for s in STRAT_ORDER}
    vision_sums = {s: 0.0 for s in STRAT_ORDER}

    for b in blobs:
        strategy = b.strategy
        if strategy not in counts:
            continue
        counts[strategy] += 1
        speed_sums[strategy] += b.speed
        vision_sums[strategy] += b.vision

    history_frames.append(frame)
    for s in STRAT_ORDER:
        ratio = counts[s] / total
        avg_speed = speed_sums[s] / counts[s] if counts[s] else 0.0
        avg_vision = vision_sums[s] / counts[s] if counts[s] else 0.0
        history_ratios[s].append(ratio)
        history_avg_speed[s].append(avg_speed)
        history_avg_vision[s].append(avg_vision)

# ======================
# Animation update
# ======================
def update(frame):
    global blobs, foods

    # Move + eat
    for b in blobs:
        b.move(foods)
        b.eat(foods)

    # Reproduce
    babies = []
    for b in blobs:
        child = b.reproduce()
        if child:
            babies.append(child)
    blobs.extend(babies)

    # Cull dead
    blobs = [b for b in blobs if b.energy > 0]

    # Limited respawn
    limited_food_respawn()

    # --- World view ---
    if blobs:
        positions = np.array([b.pos for b in blobs])
        colors = [STRAT_COLOR[b.strategy] for b in blobs]
        scat_blobs.set_offsets(positions)
        scat_blobs.set_facecolors(colors)
        scat_blobs.set_edgecolors(colors)
    else:
        scat_blobs.set_offsets(np.empty((0,2)))
        scat_blobs.set_facecolors([])
        scat_blobs.set_edgecolors([])

    if foods:
        scat_food.set_offsets(np.array(foods))
    else:
        scat_food.set_offsets(np.empty((0,2)))

    avg_speed = (np.mean([b.speed for b in blobs]) if blobs else 0.0)
    avg_vision = (np.mean([b.vision for b in blobs]) if blobs else 0.0)
    ax_world.set_xlabel(f"Frame {frame} | Pop: {len(blobs)} | Avg speed: {avg_speed:.2f} | Avg vision: {avg_vision:.1f}")

    # --- Stats tracking (ratio / avg speed / avg vision) ---
    compute_and_record_stats(frame)
    x_upper = max(100, frame + 10)
    for axis in (ax_ratio, ax_speed, ax_vision):
        axis.set_xlim(0, x_upper)

    for s in STRAT_ORDER:
        ratio_lines[s].set_data(history_frames, history_ratios[s])
        avg_speed_lines[s].set_data(history_frames, history_avg_speed[s])
        avg_vision_lines[s].set_data(history_frames, history_avg_vision[s])

    return (
        scat_blobs,
        scat_food,
        *ratio_lines.values(),
        *avg_speed_lines.values(),
        *avg_vision_lines.values(),
    )

def toggle_pause(event):
    """Toggle pause/resume when spacebar or 'p' is pressed."""
    global paused
    if event.key not in (" ", "p"):
        return
    paused = not paused
    if paused:
        ani.event_source.stop()
    else:
        ani.event_source.start()

ani = FuncAnimation(fig, update, frames=MAX_FRAMES, interval=60, blit=False, repeat=False)
fig.canvas.mpl_connect("key_press_event", toggle_pause)
plt.tight_layout()
plt.show()
