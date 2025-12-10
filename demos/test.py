# simple_maze_policy_iteration.py
# Simple Maze + Policy Iteration (model-based)
# EDIT only in the section marked: "# --- USER: edit here ---"
# Run: python simple_maze_policy_iteration.py

import numpy as np
import cv2

# --------------------------
# 1) Maze config (numpy grid)
# --------------------------
ROWS, COLS = 5, 5
WALLS = [(1,1),(1,2),(1,3),(3,1),(3,2)]   # list of blocked cells (r,c)
START = (0,0)
GOAL  = (4,4)

# Build walls array
walls = np.zeros((ROWS, COLS), dtype=np.int8)
for (r,c) in WALLS:
    walls[r,c] = 1

# Utility: state index <-> (r,c)
def rc2s(r,c): return int(r*COLS + c)
def s2rc(s): return (s // COLS, s % COLS)

N_STATES = ROWS * COLS
N_ACTIONS = 4   # 0=up,1=right,2=down,3=left

# --------------------------
# --- USER: edit here ------
# Define reward and terminal logic here. Keep it simple.
# Function signature: reward_fn(next_pos, hit_wall, step_count) -> (reward, done)
# - next_pos: (r,c) after moving (or same if hit wall)
# - hit_wall: True if attempted move hit wall or boundary
# - step_count: integer step number during rollout (if needed)
# --------------------------
def reward_fn(next_pos, hit_wall, step_count):
    # Example default:
    # +10 for reaching goal, -5 for bumping into wall, -1 per step otherwise.
    if next_pos == GOAL:
        return 10.0, True
    if hit_wall:
        return -5.0, False
    return -1.0, False
# --------------------------
# End editable section
# --------------------------

# --------------------------
# 2) Deterministic dynamics using numpy
#    next_state_reward_done(s, a, step_count)
# --------------------------
ACTION_DELTA = {0:(-1,0), 1:(0,1), 2:(1,0), 3:(0,-1)}

def next_state_reward_done(s, a, step_count=0):
    r, c = s2rc(s)
    dr, dc = ACTION_DELTA[int(a)]
    nr, nc = r + dr, c + dc
    hit_wall = False
    # boundary or wall => stay and mark hit_wall
    if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
        nr, nc = r, c
        hit_wall = True
    elif walls[nr, nc] == 1:
        nr, nc = r, c
        hit_wall = True
    next_s = rc2s(nr, nc)
    reward, done = reward_fn((nr,nc), hit_wall, step_count)
    return next_s, float(reward), bool(done)

# --------------------------
# 3) Build model P[s][a] = [(prob, next_s, r, done)]
#    (deterministic -> single tuple with prob=1)
# --------------------------
def build_model():
    P = {s: {a: [] for a in range(N_ACTIONS)} for s in range(N_STATES)}
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            ns, r, done = next_state_reward_done(s, a, step_count=0)
            P[s][a].append((1.0, ns, r, done))
    return P

# --------------------------
# 4) Policy Evaluation (model-based)
# --------------------------
def policy_evaluation(P, policy, gamma=0.95, theta=1e-6):
    V = np.zeros(N_STATES)
    while True:
        delta = 0.0
        for s in range(N_STATES):
            # skip wall cells (optional: treat wall states as not used)
            r,c = s2rc(s)
            if walls[r,c] == 1:
                continue
            v = 0.0
            for a, pi_sa in enumerate(policy[s]):
                for (prob, ns, rew, done) in P[s][a]:
                    v += pi_sa * prob * (rew + (0.0 if done else gamma * V[ns]))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

# --------------------------
# 5) Policy Iteration (model-based)
# --------------------------
def policy_iteration(P, gamma=0.95):
    policy = np.ones((N_STATES, N_ACTIONS)) / N_ACTIONS  # start uniform
    while True:
        V = policy_evaluation(P, policy, gamma=gamma)
        policy_stable = True
        for s in range(N_STATES):
            r,c = s2rc(s)
            if walls[r,c] == 1:
                policy[s] = np.zeros(N_ACTIONS)
                continue
            old_action = int(np.argmax(policy[s]))
            # compute q for each action
            q = np.zeros(N_ACTIONS)
            for a in range(N_ACTIONS):
                for (prob, ns, rew, done) in P[s][a]:
                    q[a] += prob * (rew + (0.0 if done else gamma * V[ns]))
            best_a = int(np.argmax(q))
            new_pi = np.zeros(N_ACTIONS); new_pi[best_a] = 1.0
            policy[s] = new_pi
            if best_a != old_action:
                policy_stable = False
        if policy_stable:
            return policy, V

# --------------------------
# 6) Minimal OpenCV UI to show grid, values and greedy arrows
#    Press SPACE to run a greedy rollout from START. ESC or q to quit.
# --------------------------
CELL = 80
PAD = 6
FONT = cv2.FONT_HERSHEY_SIMPLEX

def init_frame():
    H = ROWS * CELL + PAD*2
    W = COLS * CELL + PAD*2
    img = np.ones((H,W,3), dtype=np.uint8) * 255
    # grid lines
    for r in range(ROWS+1):
        y = PAD + r*CELL
        cv2.line(img, (PAD, y), (W-PAD, y), (0,0,0), 1)
    for c in range(COLS+1):
        x = PAD + c*CELL
        cv2.line(img, (x, PAD), (x, H-PAD), (0,0,0), 1)
    # walls
    for r in range(ROWS):
        for c in range(COLS):
            if walls[r,c] == 1:
                tl = (PAD + c*CELL + 2, PAD + r*CELL + 2)
                br = (PAD + (c+1)*CELL - 2, PAD + (r+1)*CELL - 2)
                cv2.rectangle(img, tl, br, (0,0,0), -1)
    # goal
    gr,gc = GOAL
    center = (PAD + gc*CELL + CELL//2, PAD + gr*CELL + CELL//2)
    cv2.circle(img, center, CELL//4, (0,150,0), -1)
    cv2.putText(img, "G", (center[0]-12, center[1]+12), FONT, 0.9, (255,255,255), 2)
    return img

def draw_values_policy(img, V, policy):
    for s in range(N_STATES):
        r,c = s2rc(s)
        if walls[r,c] == 1:
            continue
        cx = PAD + c*CELL + CELL//2
        cy = PAD + r*CELL + CELL//2
        val = V[s]
        cv2.putText(img, f"{val:.1f}", (cx-28, cy-6), FONT, 0.45, (50,50,200), 1)
        a = int(np.argmax(policy[s])) if np.sum(policy[s])>0 else None
        if a == 0:
            cv2.arrowedLine(img, (cx, cy+12), (cx, cy-12), (120,0,120), 2, tipLength=0.4)
        elif a == 1:
            cv2.arrowedLine(img, (cx-12, cy), (cx+12, cy), (120,0,120), 2, tipLength=0.4)
        elif a == 2:
            cv2.arrowedLine(img, (cx, cy-12), (cx, cy+12), (120,0,120), 2, tipLength=0.4)
        elif a == 3:
            cv2.arrowedLine(img, (cx+12, cy), (cx-12, cy), (120,0,120), 2, tipLength=0.4)
    return img

def draw_agent(img, s):
    r,c = s2rc(s)
    x = PAD + c*CELL + 12
    y = PAD + (r+1)*CELL - 18
    cv2.putText(img, "A", (x,y), FONT, 1.1, (0,0,255), 3)
    return img

# --------------------------
# 7) Run policy iteration and UI
# --------------------------
def main():
    P = build_model()
    policy, V = policy_iteration(P, gamma=0.95)

    cv2.namedWindow("Maze - Policy Iteration", cv2.WINDOW_AUTOSIZE)
    frame_base = init_frame()
    frame_shown = draw_values_policy(frame_base.copy(), V, policy)
    cv2.imshow("Maze - Policy Iteration", frame_shown)
    print("Done. Press SPACE to run greedy rollout, ESC/q to quit. Edit reward in reward_fn(...) as needed.")

    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == 27 or k == ord('q'):
            break
        if k == 32:
            # run greedy rollout from START
            s = rc2s(*START)
            done = False
            steps = 0
            max_steps = 200
            while not done and steps < max_steps:
                frame = init_frame()
                frame = draw_values_policy(frame, V, policy)
                frame = draw_agent(frame, s)
                cv2.imshow("Maze - Policy Iteration", frame)
                cv2.waitKey(200)
                a = int(np.argmax(policy[s])) if np.sum(policy[s])>0 else np.random.randint(0,N_ACTIONS)
                ns, rwd, done = next_state_reward_done(s, a, step_count=steps+1)
                s = ns
                steps += 1
            # final display
            frame = init_frame()
            frame = draw_values_policy(frame, V, policy)
            frame = draw_agent(frame, s)
            cv2.imshow("Maze - Policy Iteration", frame)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
