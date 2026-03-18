import os
import random
import warnings
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

# ML
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# RL
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN, PPO

# Stats
from scipy.stats import wilcoxon

# MILP
import pulp

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIG
# ============================================================

CONFIG = {
    # --------------------------------------------------------
    # Data + Frequencies
    # --------------------------------------------------------
    "file_path": "household_power_consumption.txt",

    # FREQUENCY CONTROL: The code iterates through these keys
    "freqs": {"Hourly": "H", "Daily": "D", "Weekly": "W"},

    # Context window for each frequency
    "time_step": {"H": 24 * 7, "D": 30, "W": 8},

    # --------------------------------------------------------
    # Forecasting
    # --------------------------------------------------------
    "max_epochs_forecast": 50,
    "forecast_patience": 5,  # Early Stopping Patience
    "batch_size": 64,
    "lr_forecast": 1e-3,

    # --------------------------------------------------------
    # RL
    # --------------------------------------------------------
    "rl_timesteps": 40000,
    "seeds": [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001],

    # --------------------------------------------------------
    # Appliance Constraints
    # --------------------------------------------------------
    "KW1": 1.5,
    "KW2": 2.0,
    "M1": 3,
    "M2": 3,

    # --------------------------------------------------------
    # Peak + Penalty
    # --------------------------------------------------------
    "PEAK": 4.5,
    "PENALTY_WEIGHT": 50.0,

    # Hybrid safety margin
    "SAFETY_BUFFER": 0.05,

    # --------------------------------------------------------
    # Hybrid Weights
    # --------------------------------------------------------
    "ALPHA_RL": 1.5,          # RL influence weight
    "BETA_COMFORT": 0.3,      # Comfort penalty weight
    "ADJ_FACTOR": 0.5,        # Block builder adjacency factor

    # --------------------------------------------------------
    # Robust MILP
    # --------------------------------------------------------
    "ROBUST_BETA": 1.28,      # ~90th percentile

    # --------------------------------------------------------
    # Multi-objective weights
    # --------------------------------------------------------
    "W_DISCOMFORT": 2.0,
    "W_SWITCH": 1.0,

    # --------------------------------------------------------
    # MILP solver options
    # --------------------------------------------------------
    "MILP_MSG": 0,
    "MILP_TIME_LIMIT": 30,

    "TOU_TARIFF": {
        "OFF_PEAK": 1.0,
        "MID_PEAK": 1.4,
        "PEAK": 2.0,
        "MID_START": 8,
        "MID_END": 18,
        "PEAK_START": 18,
        "PEAK_END": 22,
    },
}


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. TOU TARIFF
# ============================================================

def create_tou_tariff(index):
    hours = index.hour
    tariff = CONFIG["TOU_TARIFF"]

    price = np.ones(len(index), dtype=np.float32) * tariff["OFF_PEAK"]

    price[(hours >= tariff["MID_START"]) &
          (hours < tariff["MID_END"])] = tariff["MID_PEAK"]

    price[(hours >= tariff["PEAK_START"]) &
          (hours < tariff["PEAK_END"])] = tariff["PEAK"]

    return price


# ============================================================
# 4. DATA LOADING (CUSTOM SPLIT 60/20/20)
# ============================================================

def get_uci_data(freq="H", file_path="household_power_consumption.txt"):
    """
    Loads, preprocesses, and prepares the UCI dataset.
    Auto-selects time_step based on freq: H=60, D=30, W=8.
    """
    freq = freq.upper()

    # --- AUTO-SELECT TIME STEP ---
    if freq == 'W':
        time_step = 8
    elif freq == 'D':
        time_step = 30
    else:
        time_step = 60

    print(f"--- Loading UCI dataset with frequency: {freq} and time_step: {time_step} ---")

    if not os.path.exists(file_path):
        print(f"[INFO] File not found. Using SYNTHETIC data for freq={freq} ...")
        periods = 24 * 365 * 2 if freq == "H" else (365 * 2 if freq == "D" else 104)
        dates = pd.date_range(start="2010-01-01", periods=periods, freq=freq)
        x = np.linspace(0, 50, len(dates))
        data = (1.5 + np.sin(x) + 0.4 * np.sin(6 * x)) + np.random.normal(0, 0.15, len(dates))
        data = np.clip(data, 0, None)
        df_resampled = pd.DataFrame(data, index=dates, columns=["Global_active_power"])
        df_resampled["price"] = create_tou_tariff(dates)
    else:
        df = pd.read_csv(
            file_path,
            sep=';',
            low_memory=False,
            na_values=['?', 'nan', 'NaN']
        )

        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
        df = df.set_index('DateTime')
        df.drop(['Date', 'Time'], axis=1, inplace=True, errors='ignore')

        df = df.apply(pd.to_numeric, errors='coerce')
        power_column = 'Global_active_power'

        df = df[[power_column]].dropna()
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        df_resampled = df[power_column].resample(freq).mean().to_frame()
        df_resampled.ffill(inplace=True)
        df_resampled.bfill(inplace=True)

        df_resampled["price"] = create_tou_tariff(df_resampled.index)

    # --- SPLITTING LOGIC (60/20/20) ---
    data_values = df_resampled["Global_active_power"].values.reshape(-1, 1)
    price_values = df_resampled["price"].values
    index_values = df_resampled.index

    total_len = len(data_values)
    train_size = int(total_len * 0.6)
    val_size = int(total_len * 0.2)
    # Remaining is test (~20%)

    # Slicing Data
    train_data = data_values[:train_size]
    val_data = data_values[train_size : train_size + val_size]
    test_data = data_values[train_size + val_size:]

    # Slicing Price & Index
    test_price = price_values[train_size + val_size:]
    index_test = index_values[train_size + val_size:]

    # Check length
    if len(train_data) <= time_step or len(val_data) <= time_step or len(test_data) <= time_step:
        print("!!! Not enough data for splits. !!!")
        return None, None, None, None

    # --- SCALING ---
    scaler = MinMaxScaler()
    scaler.fit(train_data)

    scaled_train = scaler.transform(train_data)
    scaled_val = scaler.transform(val_data)
    scaled_test = scaler.transform(test_data)

    # --- SEQUENCING ---
    def create_sequences(data, ts):
        X, Y = [], []
        for i in range(len(data) - ts):
            X.append(data[i:(i + ts), 0])
            Y.append(data[i + ts, 0])
        return np.array(X), np.array(Y)

    X_train, y_train = create_sequences(scaled_train, time_step)
    X_val, y_val = create_sequences(scaled_val, time_step)
    X_test, y_test = create_sequences(scaled_test, time_step)

    # Reshape
    X_train = X_train.reshape(-1, time_step, 1)
    X_val = X_val.reshape(-1, time_step, 1)
    X_test = X_test.reshape(-1, time_step, 1)

    # Tensor Datasets
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.float32))

    extra = {
        "scaler": scaler,
        "test_price": test_price[time_step:],
        "index_test": index_test[time_step:],
    }

    return train_ds, val_ds, test_ds, extra


# ============================================================
# 5. PROBABILISTIC FORECASTING (DeepAR-Style)
# ============================================================

class DeepARStyle(nn.Module):
    def __init__(self, hidden=64, num_layers=1, dropout=0.0):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc_mu = nn.Linear(hidden, 1)
        self.fc_sigma = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]

        mu = self.fc_mu(last)
        sigma = torch.nn.functional.softplus(self.fc_sigma(last)) + 1e-6

        return mu.squeeze(-1), sigma.squeeze(-1)

def gaussian_nll(mu, sigma, y):
    return torch.mean(
        torch.log(sigma) + (y - mu) ** 2 / (2.0 * sigma ** 2)
    )


def forecast_probabilistic(model, ds):
    device = next(model.parameters()).device
    X = ds.tensors[0].to(device)

    model.eval()
    with torch.no_grad():
        mu, sigma = model(X)

    return mu.cpu().numpy(), sigma.cpu().numpy()


# ============================================================
# 6. STRICT BLOCK BUILDER
# ============================================================
def build_strict_schedule(score_vec, H_target, min_on):
    adj_factor = CONFIG["ADJ_FACTOR"]

    score_vec = np.asarray(score_vec, dtype=float)
    T = len(score_vec)
    A = np.zeros(T, dtype=int)
    H_target = int(H_target)
    min_on = max(1, int(min_on))

    if T == 0 or H_target <= 0:
        return A

    H_target = min(H_target, T)

    # 1. Safety Masks
    FORBIDDEN = np.isinf(score_vec) | np.isnan(score_vec) | (score_vec >= 1e8)

    # 2. Stable Adjacency Bonus (Robust Version)
    finite_scores = score_vec[~FORBIDDEN]
    if len(finite_scores) == 0:
        return A

    score_std = float(np.std(finite_scores) + 1e-6)
    ADJACENCY_BONUS = adj_factor * score_std

    current_h = 0

    # =========================================================
    # Phase 1: Place Minimum Blocks
    # =========================================================
    while current_h + min_on <= H_target:
        best_val = float("inf")
        best_start = -1

        for t in range(T - min_on + 1):

            if np.any(A[t:t + min_on] == 1):
                continue
            if np.any(FORBIDDEN[t:t + min_on]):
                continue

            block_score = float(np.sum(score_vec[t:t + min_on]))

            # Adjacency encouragement
            is_adj = False
            if t > 0 and A[t - 1] == 1:
                is_adj = True
            if t + min_on < T and A[t + min_on] == 1:
                is_adj = True

            if is_adj:
                block_score -= ADJACENCY_BONUS

            if block_score < best_val:
                best_val = block_score
                best_start = t

        if best_start == -1:
            break

        A[best_start:best_start + min_on] = 1
        current_h += min_on

    # =========================================================
    # Phase 2: Extend Smoothly
    # =========================================================
    while current_h < H_target:
        best_val = float("inf")
        best_idx = -1

        for t in range(T):
            if A[t] == 1 or FORBIDDEN[t]:
                continue

            val = float(score_vec[t])

            is_adj = (t > 0 and A[t - 1] == 1) or \
                     (t < T - 1 and A[t + 1] == 1)

            if is_adj:
                val -= ADJACENCY_BONUS

            if val < best_val:
                best_val = val
                best_idx = t

        if best_idx == -1:
            break

        A[best_idx] = 1
        current_h += 1

    # =========================================================
    # Phase 3: Emergency Fill
    # =========================================================
    if current_h < H_target:
        remaining = np.where((A == 0) & (~FORBIDDEN))[0]
        if len(remaining) > 0:
            rem_sorted = remaining[np.argsort(score_vec[remaining])]
            need = H_target - current_h
            A[rem_sorted[:need]] = 1

    return A


# ============================================================
# 7. MILP MODELS (Full Comparison Suite)
# ============================================================

def _build_and_solve_milp(load_input, price, PEAK, KW1, KW2,
                          H1, H2, M1, M2,
                          penalty_weight,
                          strict_peak=False):
    """
    Internal MILP builder.
    """

    T = len(load_input)
    prob = pulp.LpProblem("TwoAppliance_MILP", pulp.LpMinimize)

    # Decision variables
    u1 = pulp.LpVariable.dicts("u1", range(T), cat="Binary")
    u2 = pulp.LpVariable.dicts("u2", range(T), cat="Binary")
    s1 = pulp.LpVariable.dicts("s1", range(T), cat="Binary")
    s2 = pulp.LpVariable.dicts("s2", range(T), cat="Binary")

    if not strict_peak:
        viol = pulp.LpVariable.dicts("viol", range(T), lowBound=0)

    # Total ON hours
    prob += pulp.lpSum([u1[t] for t in range(T)]) == H1
    prob += pulp.lpSum([u2[t] for t in range(T)]) == H2

    # Minimum run constraints
    for t in range(T):
        ks1 = [k for k in range(max(0, t - M1 + 1), t + 1)]
        ks2 = [k for k in range(max(0, t - M2 + 1), t + 1)]
        prob += u1[t] <= pulp.lpSum([s1[k] for k in ks1])
        prob += u2[t] <= pulp.lpSum([s2[k] for k in ks2])

    for k in range(T):
        for j in range(M1):
            if k + j < T:
                prob += u1[k + j] >= s1[k]
        for j in range(M2):
            if k + j < T:
                prob += u2[k + j] >= s2[k]

    # Peak constraints
    for t in range(T):
        total_load = float(load_input[t]) + KW1 * u1[t] + KW2 * u2[t]

        if strict_peak:
            prob += total_load <= PEAK
        else:
            prob += total_load <= PEAK + viol[t]

    # Objective
    bill = pulp.lpSum([
        (float(load_input[t]) + KW1 * u1[t] + KW2 * u2[t]) * float(price[t])
        for t in range(T)
    ])

    if strict_peak:
        prob += bill
    else:
        penalty = penalty_weight * pulp.lpSum([viol[t] for t in range(T)])
        prob += bill + penalty

    # Solve
    solver = pulp.PULP_CBC_CMD(
        msg=CONFIG["MILP_MSG"],
        timeLimit=CONFIG["MILP_TIME_LIMIT"]
    )
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]

    if status != "Optimal":
        print(f"[MILP] Warning: status={status} -> fallback zeros")
        return np.zeros(T, dtype=int), np.zeros(T, dtype=int), status

    A1 = np.array([int(round(pulp.value(u1[t]))) for t in range(T)], dtype=int)
    A2 = np.array([int(round(pulp.value(u2[t]))) for t in range(T)], dtype=int)

    return A1, A2, status


# ------------------------------------------------------------
# 7.1 Pointwise Deterministic MILP (Perfect Info)
# ------------------------------------------------------------

def solve_milp_pointwise(y_true, price, PEAK,
                         KW1, KW2, H1, H2, M1, M2,
                         penalty_weight):
    return _build_and_solve_milp(
        load_input=y_true,
        price=price,
        PEAK=PEAK,
        KW1=KW1,
        KW2=KW2,
        H1=H1,
        H2=H2,
        M1=M1,
        M2=M2,
        penalty_weight=penalty_weight,
        strict_peak=False
    )


# ------------------------------------------------------------
# 7.2 Forecast-Based Deterministic MILP
# ------------------------------------------------------------

def solve_milp_forecast(y_forecast, price, PEAK,
                        KW1, KW2, H1, H2, M1, M2,
                        penalty_weight):
    return _build_and_solve_milp(
        load_input=y_forecast,
        price=price,
        PEAK=PEAK,
        KW1=KW1,
        KW2=KW2,
        H1=H1,
        H2=H2,
        M1=M1,
        M2=M2,
        penalty_weight=penalty_weight,
        strict_peak=False
    )


# ------------------------------------------------------------
# 7.3 Robust Quantile MILP
# ------------------------------------------------------------

def solve_milp_robust(mu, sigma, price, PEAK,
                      KW1, KW2, H1, H2, M1, M2,
                      penalty_weight,
                      beta=1.28):
    y_robust = mu + beta * sigma

    return _build_and_solve_milp(
        load_input=y_robust,
        price=price,
        PEAK=PEAK,
        KW1=KW1,
        KW2=KW2,
        H1=H1,
        H2=H2,
        M1=M1,
        M2=M2,
        penalty_weight=penalty_weight,
        strict_peak=False
    )


# ------------------------------------------------------------
# 7.4 Strict-Peak MILP (No Violations Allowed)
# ------------------------------------------------------------

def solve_milp_strict_peak(y_input, price, PEAK,
                           KW1, KW2, H1, H2, M1, M2):
    return _build_and_solve_milp(
        load_input=y_input,
        price=price,
        PEAK=PEAK,
        KW1=KW1,
        KW2=KW2,
        H1=H1,
        H2=H2,
        M1=M1,
        M2=M2,
        penalty_weight=0.0,
        strict_peak=True
    )


# ============================================================
# 8. RL ENV (Unchanged)
# ============================================================

class TwoApplianceEnv(gym.Env):
    """
    action=0: none
    action=1: appliance1
    action=2: appliance2
    action=3: both
    """
    def __init__(self, y, p, peak, kw1, kw2, H1, H2, m1, m2):
        super().__init__()
        self.y = np.array(y, dtype=np.float32)
        self.p = np.array(p, dtype=np.float32)
        self.T = len(self.y)

        self.peak = float(peak)
        self.kw1, self.kw2 = float(kw1), float(kw2)
        self.H1, self.H2 = int(H1), int(H2)
        self.m1, self.m2 = int(m1), int(m2)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.on1, self.on2 = 0, 0
        self.lock1, self.lock2 = 0, 0
        return self._obs(), {}

    def _obs(self):
        if self.t >= self.T:
            return np.zeros(5, dtype=np.float32)
        rem1 = (self.H1 - self.on1) / max(1, self.H1)
        rem2 = (self.H2 - self.on2) / max(1, self.H2)
        lock = (self.lock1 > 0) * 1.0 + (self.lock2 > 0) * 2.0
        return np.array([self.y[self.t], self.p[self.t], rem1, rem2, lock], dtype=np.float32)

    def step(self, action):
        w1 = 1 if action in [1, 3] else 0
        w2 = 1 if action in [2, 3] else 0

        if self.lock1 > 0:
            w1 = 1
        if self.lock2 > 0:
            w2 = 1

        steps_left = self.T - self.t
        if (self.H1 - self.on1) >= steps_left:
            w1 = 1
        if (self.H2 - self.on2) >= steps_left:
            w2 = 1

        if self.on1 >= self.H1:
            w1 = 0
        if self.on2 >= self.H2:
            w2 = 0

        if self.lock1 == 0 and w1 == 1:
            self.lock1 = self.m1
        if self.lock2 == 0 and w2 == 1:
            self.lock2 = self.m2

        if self.t >= self.T:
            return self._obs(), 0.0, True, False, {"a1": 0, "a2": 0}

        load = self.y[self.t] + self.kw1 * w1 + self.kw2 * w2
        cost = float(self.p[self.t] * load)
        viol = max(0.0, float(load - self.peak))

        reward = -cost - 50.0 * viol

        self.on1 += w1
        self.on2 += w2
        self.t += 1

        if self.lock1 > 0:
            self.lock1 -= 1
        if self.lock2 > 0:
            self.lock2 -= 1

        done = self.t >= self.T
        return self._obs(), reward, done, False, {"a1": w1, "a2": w2}


# ============================================================
# 9. RL TRAINING VARIANTS
# ============================================================

def train_dqn(env_fn, timesteps, seed):
    model = DQN(
        "MlpPolicy",
        env_fn(),
        verbose=0,
        learning_rate=5e-4,
        buffer_size=50000,
        gamma=0.99,
        seed=seed
    )
    model.learn(total_timesteps=timesteps)
    return model


def train_ppo(env_fn, timesteps, seed):
    model = PPO(
        "MlpPolicy",
        env_fn(),
        verbose=0,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        seed=seed
    )
    model.learn(total_timesteps=timesteps)
    return model


# ------------------------------------------------------------
# Risk-Sensitive Environment (Stronger Peak Penalty)
# ------------------------------------------------------------

class RiskAwareEnv(TwoApplianceEnv):
    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)

        # amplify violation penalty
        load = self.y[self.t - 1] + self.kw1 * info["a1"] + self.kw2 * info["a2"]
        viol = max(0.0, float(load - self.peak))

        reward -= 200.0 * viol  # stronger risk aversion

        return obs, reward, done, trunc, info


def train_risk_dqn(env_fn, timesteps, seed):
    model = DQN(
        "MlpPolicy",
        env_fn(),
        verbose=0,
        learning_rate=5e-4,
        buffer_size=50000,
        gamma=0.99,
        seed=seed
    )
    model.learn(total_timesteps=timesteps)
    return model


# ============================================================
# 10. ROLLOUT FUNCTION
# ============================================================

def rollout_schedule(model, env_fn):
    env = env_fn()
    obs, _ = env.reset()
    A1, A2 = [], []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(int(action))
        A1.append(info.get("a1", 0))
        A2.append(info.get("a2", 0))

    return np.array(A1, dtype=int), np.array(A2, dtype=int)


# ============================================================
# 10. HYBRID PRINCIPLE (Upgraded Structured Version)
# ============================================================

def get_rl_q_values_for_actions(rl_model, env):
    """
    Extract Q-values along greedy rollout trajectory.
    Works for DQN-based models.
    """
    obs, _ = env.reset()
    done = False
    q_list = []

    # FIX: SB3 DQN stores q_net inside policy
    q_net = rl_model.policy.q_net
    device = rl_model.device

    while not done:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            q_vals = q_net(obs_tensor)

        q_list.append(q_vals.detach().cpu().numpy()[0])

        action = int(q_vals.argmax().item())
        obs, _, done, _, _ = env.step(action)

    return np.array(q_list)


def hybrid_score_improved(price,
                          mu_forecast,
                          sigma_forecast,
                          q_vals,
                          peak,
                          kw,
                          which_app,
                          index):

    eps = 1e-6

    alpha_rl = CONFIG["ALPHA_RL"]
    beta_comfort = CONFIG["BETA_COMFORT"]
    safety_buffer = CONFIG["SAFETY_BUFFER"]
    beta_risk = CONFIG["ROBUST_BETA"]

    # 1. Economic
    price_norm = (price - price.mean()) / (price.std() + eps)

    # 2. RL
    if which_app == 1:
        q_on = np.maximum(q_vals[:, 1], q_vals[:, 3])
    else:
        q_on = np.maximum(q_vals[:, 2], q_vals[:, 3])

    q_norm = (q_on - q_on.mean()) / (q_on.std() + eps)
    score_rl = -q_norm

    # 3. Risk Projection
    projected = mu_forecast + beta_risk * sigma_forecast + kw
    violation_mask = np.where(projected > (peak - safety_buffer), 1e9, 0.0)

    # 4. Comfort
    hours = index.hour.values

    if which_app == 1:
        preferred = (hours >= 6) & (hours < 10)
    else:
        preferred = (hours >= 18) & (hours < 22)

    comfort_penalty = np.where(~preferred, 1.0, 0.0)

    # Final
    final_score = (
        price_norm
        + alpha_rl * score_rl
        + violation_mask
        + beta_comfort * comfort_penalty
    )

    return final_score


# ============================================================
# 11. METRICS (Forecast + Scheduling)
# ============================================================

# ------------------------------------------------------------
# 11.1 Forecasting Metrics
# ------------------------------------------------------------

def evaluate_forecast(y_true, y_pred):
    """
    Standard forecasting metrics.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(
        np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100.0
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_%": mape
    }


# ------------------------------------------------------------
# 11.2 Comfort Metrics
# ------------------------------------------------------------

def compute_discomfort(index, A1, A2):
    if len(index) == 0:
        return 0.0

    hours = index.hour.values
    pref1 = (hours >= 6) & (hours < 10)
    pref2 = (hours >= 18) & (hours < 22)

    dis1 = np.sum((A1 == 1) & (~pref1))
    dis2 = np.sum((A2 == 1) & (~pref2))

    return float(dis1 + dis2)


def compute_switches(A):
    if len(A) <= 1:
        return 0.0
    return float(np.sum(np.abs(np.diff(A))))


# ------------------------------------------------------------
# 11.3 Full Scheduling Evaluation
# ------------------------------------------------------------

def evaluate_schedule_full(y_true,
                           price,
                           idx,
                           A1,
                           A2,
                           KW1,
                           KW2,
                           PEAK,
                           penalty_weight):

    # --------------------------------------------------------
    # Final Load
    # --------------------------------------------------------
    load = y_true + KW1 * A1 + KW2 * A2

    # --------------------------------------------------------
    # Economic Cost
    # --------------------------------------------------------
    bill = float(np.sum(load * price))

    # --------------------------------------------------------
    # Peak Violations
    # --------------------------------------------------------
    viol_vec = np.maximum(0.0, load - PEAK)
    viol_sum = float(np.sum(viol_vec))
    viol_rate = float(np.mean(viol_vec > 0.0))

    # --------------------------------------------------------
    # Peak Statistics
    # --------------------------------------------------------
    peak_load = float(np.max(load))
    baseline_peak = float(np.max(y_true))
    peak_reduction = float(
        (baseline_peak - peak_load) / (baseline_peak + 1e-6)
    )

    # --------------------------------------------------------
    # Load Smoothness
    # --------------------------------------------------------
    load_variance = float(np.var(load))

    # --------------------------------------------------------
    # Comfort & Switching
    # --------------------------------------------------------
    discomfort = compute_discomfort(idx, A1, A2)
    switches = compute_switches(A1) + compute_switches(A2)

    # --------------------------------------------------------
    # Composite Objectives
    # --------------------------------------------------------
    milp_total = bill + penalty_weight * viol_sum

    mo_score = (
        milp_total
        + CONFIG["W_DISCOMFORT"] * discomfort
        + CONFIG["W_SWITCH"] * switches
    )

    return {
        "Bill": bill,
        "ViolSum": viol_sum,
        "ViolRate": viol_rate,
        "PeakLoad": peak_load,
        "PeakReduction": peak_reduction,
        "LoadVariance": load_variance,
        "Discomfort": discomfort,
        "Switches": switches,
        "MILP_Total": milp_total,
        "MO_Score": mo_score
    }


# ------------------------------------------------------------
# 11.4 Regret Utility
# ------------------------------------------------------------

def safe_regret(value, oracle_value):
    return float(
        100.0 * (value - oracle_value) / (abs(oracle_value) + 1e-6)
    )

# ============================================================
# 12. STATISTICAL ANALYSIS
# ============================================================

def rank_biserial_effect(x, y):
    x = np.array(x)
    y = np.array(y)
    d = x - y
    d = d[d != 0]
    n = len(d)

    if n == 0:
        return 0.0

    ranks = pd.Series(np.abs(d)).rank().values
    Wpos = np.sum(ranks[d > 0])

    return float(1.0 - (2.0 * Wpos) / (n * (n + 1)))


def interpret_effect_size(rbc):
    r = abs(rbc)

    if r < 0.1:
        return "Negligible"
    elif r < 0.3:
        return "Small"
    elif r < 0.5:
        return "Medium"
    else:
        return "Large"


def wilcoxon_report(df, methodA, methodB, metric):
    dfA = df[df["Method"] == methodA][["Seed", metric]]
    dfB = df[df["Method"] == methodB][["Seed", metric]]

    merged = pd.merge(dfA, dfB, on="Seed", suffixes=("_A", "_B"))

    A = merged[f"{metric}_A"].values
    B = merged[f"{metric}_B"].values

    if len(A) < 5:
        print(f"[SKIPPED] {metric}: Not enough data points ({len(A)}) for {methodA} vs {methodB}")
        return {
            "metric": metric,
            "A": methodA,
            "B": methodB,
            "p_value": np.nan,
            "effect_size_rbc": np.nan,
            "effect_interpretation": "Insufficient data",
            "median_diff": np.nan
        }

    try:
        stat = wilcoxon(A, B, zero_method="wilcox", alternative="two-sided")
        pval = float(stat.pvalue)
    except ValueError as e:
        print(f"[ERROR] {metric}: {e}")
        pval = np.nan

    rbc = rank_biserial_effect(A, B)
    interpretation = interpret_effect_size(rbc)
    median_diff = float(np.median(A - B))

    return {
        "metric": metric,
        "A": methodA,
        "B": methodB,
        "p_value": pval,
        "effect_size_rbc": rbc,
        "effect_interpretation": interpretation,
        "median_diff": median_diff
    }

# ============================================================
# 13. PUBLICATION-QUALITY PLOTS
# ============================================================

import seaborn as sns
sns.set_style("whitegrid")


def plot_box(df, metric, title):
    plt.figure(figsize=(8, 4))
    ax = sns.boxplot(data=df, x="Method", y=metric, showfliers=False)
    sns.stripplot(data=df, x="Method", y=metric,
                  color="black", alpha=0.4, size=3)
    means = df.groupby("Method")[metric].mean().values
    ax.scatter(range(len(means)), means,
               color="red", marker="D", s=50, label="Mean")
    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=20)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_bar(summary, metric, title):
    plt.figure(figsize=(8, 4))
    means = summary[metric]["mean"]
    stds = summary[metric]["std"]
    methods = means.index.tolist()
    plt.bar(methods, means.values, yerr=stds.values, capsize=5, alpha=0.85)
    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=20)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pareto(df, title):
    plt.figure(figsize=(6, 5))
    for method in df["Method"].unique():
        sub = df[df["Method"] == method]
        if method == "Hybrid":
            plt.scatter(sub["Bill"], sub["ViolSum"],
                        label=method, s=80, edgecolor="black")
        else:
            plt.scatter(sub["Bill"], sub["ViolSum"],
                        label=method, alpha=0.6)
    plt.title(title)
    plt.xlabel("Bill")
    plt.ylabel("Violation Sum")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 14. MAIN EXPERIMENT (Full Structured Version)
# ============================================================

def run_experiment_for_frequency(freq_name, freq_code):

    freq_code = freq_code.upper()

    print(f"\n{'=' * 25} RUNNING: {freq_name} ({freq_code}) {'=' * 25}")

    # --- FIX IS HERE: Removed time_step=ts ---
    train_ds, val_ds, test_ds, extra = get_uci_data(
        freq=freq_code,
        file_path=CONFIG["file_path"]
    )

    scaler = extra["scaler"]
    price = extra["test_price"]
    idx_test = extra["index_test"]

    y_true = scaler.inverse_transform(
        test_ds.tensors[1].numpy().reshape(-1, 1)
    ).ravel()

    T = min(len(y_true), len(price), len(idx_test))
    y_true, price, idx_test = y_true[:T], price[:T], idx_test[:T]

    H1 = int(0.2 * T)
    H2 = int(0.2 * T)

    KW1, KW2 = CONFIG["KW1"], CONFIG["KW2"]
    M1, M2 = CONFIG["M1"], CONFIG["M2"]
    PEAK, PEN = CONFIG["PEAK"], CONFIG["PENALTY_WEIGHT"]

    print(f"[Setup] Steps={T}, H1={H1}, H2={H2}, PEAK={PEAK}")

    all_rows = []
    forecast_results = []

    for seed in CONFIG["seeds"]:
        print(f"\n-------------------- Seed {seed} --------------------")
        set_seed(seed)

        # =====================================================
        # 1. Forecasting (WITH EARLY STOPPING)
        # =====================================================
        print("[Forecast] Training DeepAR-style probabilistic model with Early Stopping...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = DeepARStyle(hidden=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr_forecast"])

        # Early Stopping Variables
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(CONFIG["max_epochs_forecast"]):
            # Train Loop
            model.train()
            train_losses = []
            for xb, yb in DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True):
                xb, yb = xb.to(device), yb.to(device)

                mu, sigma = model(xb)
                loss = gaussian_nll(mu, sigma, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # Validation Loop
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in DataLoader(val_ds, batch_size=CONFIG["batch_size"]):
                    xb, yb = xb.to(device), yb.to(device)
                    mu, sigma = model(xb)
                    val_loss = gaussian_nll(mu, sigma, yb)
                    val_losses.append(val_loss.item())

            avg_val_loss = np.mean(val_losses)

            # Early Stopping Check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= CONFIG["forecast_patience"]:
                    print(f"[Forecast] Early stopping at epoch {epoch}, Best Val Loss: {best_val_loss:.4f}")
                    break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        mu_test, sigma_test = forecast_probabilistic(model, test_ds)

        forecast_metrics = evaluate_forecast(
            y_true[:len(mu_test)],
            mu_test
        )
        print("[Forecast Metrics]", forecast_metrics)
        forecast_results.append({
            "Frequency": freq_name,
            "Seed": seed,
            "MAE": forecast_metrics["MAE"],
            "RMSE": forecast_metrics["RMSE"],
            "MAPE_%": forecast_metrics["MAPE_%"]
        })

        # =====================================================
        # 2. Baselines
        # =====================================================

        # Greedy
        idx_sorted = np.argsort(price)
        A1_g = np.zeros(T, dtype=int)
        A2_g = np.zeros(T, dtype=int)
        A1_g[idx_sorted[:H1]] = 1
        A2_g[idx_sorted[:H2]] = 1

        # PriceBlock
        A1_pb = build_strict_schedule(price, H1, M1)
        A2_pb = build_strict_schedule(price, H2, M2)

        # MILP Variants
        print("[MILP] Solving deterministic...")
        A1_m_det, A2_m_det, _ = solve_milp_pointwise(
            y_true, price, PEAK, KW1, KW2, H1, H2, M1, M2, PEN
        )

        print("[MILP] Solving forecast-based...")
        A1_m_for, A2_m_for, _ = solve_milp_forecast(
            mu_test[:T], price, PEAK, KW1, KW2, H1, H2, M1, M2, PEN
        )

        print("[MILP] Solving robust...")
        A1_m_rob, A2_m_rob, _ = solve_milp_robust(
            mu_test[:T], sigma_test[:T], price,
            PEAK, KW1, KW2, H1, H2, M1, M2,
            PEN,
            beta=CONFIG["ROBUST_BETA"]
        )

        # =====================================================
        # 3. RL Baselines
        # =====================================================
        def make_env(y_data=None):
            yd = y_data if y_data is not None else y_true
            return TwoApplianceEnv(yd, price, PEAK, KW1, KW2, H1, H2, M1, M2)

        def make_risk_env(y_data=None):
            yd = y_data if y_data is not None else y_true
            return RiskAwareEnv(yd, price, PEAK, KW1, KW2, H1, H2, M1, M2)

        print("[DQN] Training...")
        dqn_model = train_dqn(make_env, CONFIG["rl_timesteps"], seed)
        A1_dqn, A2_dqn = rollout_schedule(dqn_model, make_env)

        print("[PPO] Training...")
        ppo_model = train_ppo(make_env, CONFIG["rl_timesteps"], seed)
        A1_ppo, A2_ppo = rollout_schedule(ppo_model, make_env)

        print("[Risk-DQN] Training...")
        risk_model = train_risk_dqn(make_risk_env, CONFIG["rl_timesteps"], seed)
        A1_risk, A2_risk = rollout_schedule(risk_model, make_risk_env)

        # =====================================================
        # 4. Hybrid
        # =====================================================
        print("[Hybrid] Starting scheduling...")

        env_h1 = make_env(y_true)
        q_vals_1 = get_rl_q_values_for_actions(dqn_model, env_h1)

        score2 = hybrid_score_improved(
            price,
            mu_test[:T],
            sigma_test[:T],
            q_vals_1,
            PEAK,
            KW2,
            2,
            idx_test
        )

        A2_h = build_strict_schedule(score2, H2, M2)

        y_updated = y_true + (A2_h * KW2)

        env_h2 = make_env(y_updated)
        q_vals_2 = get_rl_q_values_for_actions(dqn_model, env_h2)

        score1 = hybrid_score_improved(
            price,
            mu_test[:T] + A2_h * KW2,   # updated forecast mean
            sigma_test[:T],
            q_vals_2,
            PEAK,
            KW1,
            1,
            idx_test
        )
        A1_h = build_strict_schedule(score1, H1, M1)

        # =====================================================
        # 5. Collect All Methods
        # =====================================================

        methods = {
            "Greedy": (A1_g, A2_g),
            "PriceBlock": (A1_pb, A2_pb),
            "MILP_Det": (A1_m_det, A2_m_det),
            "MILP_Robust": (A1_m_rob, A2_m_rob),
            "MILP_Forecast": (A1_m_for, A2_m_for),
            "DQN": (A1_dqn, A2_dqn),
            "PPO": (A1_ppo, A2_ppo),
            "RiskDQN": (A1_risk, A2_risk),
            "Hybrid": (A1_h, A2_h),
        }

        for method, (a1, a2) in methods.items():
            metrics = evaluate_schedule_full(
                y_true, price, idx_test,
                a1, a2,
                KW1, KW2,
                PEAK, PEN
            )
            metrics.update({
                "Frequency": freq_name,
                "Seed": seed,
                "Method": method
            })
            all_rows.append(metrics)

    # =========================================================
    # 6. Aggregation
    # =========================================================

    df = pd.DataFrame(all_rows)
    metrics_to_summarize = [
        "Bill",
        "ViolSum",
        "ViolRate",
        "PeakLoad",
        "PeakReduction",
        "LoadVariance",
        "Discomfort",
        "Switches",
        "MILP_Total",
        "MO_Score"
    ]

    summary = (
        df.groupby("Method")[metrics_to_summarize]
        .agg(["mean", "std"])
    )

    print("\n================ FINAL SUMMARY ================")
    print(summary[[
        "Bill",
        "ViolSum",
        "ViolRate",
        "PeakLoad",
        "PeakReduction",
        "LoadVariance",
        "Discomfort",
        "Switches",
        "MILP_Total",
        "MO_Score"
    ]])

    # =========================================================
    # 7. Statistical Tests (Hybrid vs Others)
    # =========================================================

    stats_rows = []
    compare_against = ["MILP_Det", "DQN", "PPO", "RiskDQN"]

    for other in compare_against:
        for metric in ["Bill", "MO_Score", "PeakReduction"]:
            stats_rows.append(
                wilcoxon_report(
                    df[df["Method"].isin(["Hybrid", other])],
                    "Hybrid",
                    other,
                    metric
                )
            )

    df_stats = pd.DataFrame(stats_rows)
    df_forecast = pd.DataFrame(forecast_results)

    print("\n================ FORECAST SUMMARY ================")
    print(
        df_forecast.groupby("Frequency")[["MAE", "RMSE", "MAPE_%"]]
        .agg(["mean", "std"])
    )

    print("\n================ WILCOXON TESTS ================")
    if not df_stats.empty:
        print(df_stats.to_string(index=False))

    return df, df_stats, df_forecast


# ============================================================
# 15. MAIN ENTRY POINT
# ============================================================

def main():

    out_dir = "results_publishable"
    os.makedirs(out_dir, exist_ok=True)

    all_df = []
    all_stats = []
    all_forecasts = []

    print("\n================ STARTING FULL EXPERIMENT ================")

    for freq_name, freq_code in CONFIG["freqs"].items():

        print(f"\n######## Frequency: {freq_name} ########")

        df, st, df_forecast = run_experiment_for_frequency(freq_name, freq_code)

        all_df.append(df)
        all_stats.append(st)
        all_forecasts.append(df_forecast)

        df.to_csv(os.path.join(out_dir, f"results_{freq_name}.csv"), index=False)
        st.to_csv(os.path.join(out_dir, f"wilcoxon_{freq_name}.csv"), index=False)

    # --------------------------------------------------------
    # Aggregate across frequencies
    # --------------------------------------------------------

    full = pd.concat(all_df, ignore_index=True)
    full_stats = pd.concat(all_stats, ignore_index=True)
    full_forecast = pd.concat(all_forecasts, ignore_index=True)

    full.to_csv(os.path.join(out_dir, "results_ALL.csv"), index=False)
    full_stats.to_csv(os.path.join(out_dir, "wilcoxon_ALL.csv"), index=False)
    full_forecast.to_csv(os.path.join(out_dir, "forecast_results_ALL.csv"), index=False)

    print("\n==================== SAVED OUTPUT FILES ====================")
    print("Saved outputs to folder:", out_dir)
    print("   - results_ALL.csv")
    print("   - wilcoxon_ALL.csv")

    for freq_name in CONFIG["freqs"]:
        print(f"   - results_{freq_name}.csv")
        print(f"   - wilcoxon_{freq_name}.csv")

    print("\n==================== EXPERIMENT COMPLETE ====================")

if __name__ == "__main__":
    main()
