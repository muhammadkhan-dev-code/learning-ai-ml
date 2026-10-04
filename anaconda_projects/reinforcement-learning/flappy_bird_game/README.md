# 🐦 Flappy Bird Deep Q-Learning (DQN) Agent

A Reinforcement Learning project that trains a Deep Q-Network (DQN) agent using PyTorch and Gymnasium to autonomously play and master the Flappy Bird game.

---

## 📁 Project Structure

```text
flappy_bird_game/
├── agent.py               # Main DQN Agent: training loop, action selection, and optimization
├── dqn.py                 # PyTorch Neural Network architecture for Q-value approximation
├── experience_replay.py   # Experience Replay Buffer (deque-based memory)
├── main.py                # Manual playable version of Flappy Bird using Pygame keyboard inputs
├── parameters.yaml        # Hyperparameter configuration file
├── requirements.txt       # Project dependencies
├── runs/                  # Directory where model checkpoints (.pt) and logs (.log) are saved
├── .gitignore             # Git ignore patterns
└── README.md              # Setup and execution guide
```

---

## 🛠️ Prerequisites & Setup

### 1. Python Version Compatibility
- **Recommended**: **Python 3.10, 3.11, or 3.12** (Standard for PyTorch & Gymnasium RL workflows).
- **Python 3.13 / 3.14**: Fully supported using `pygame-ce` (Pygame Community Edition) specified in `requirements.txt`.

### 2. Create and Activate a Virtual Environment

#### Using `venv` (Standard Python):
```bash
# Create virtual environment (Python 3.10 - 3.12 recommended)
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\activate

# Activate on Windows (Command Prompt)
.\venv\Scripts\activate.bat

# Activate on macOS / Linux
source venv/bin/activate
```

#### Using `conda` (Anaconda / Miniconda):
```bash
# Create a conda environment with Python 3.11
conda create -n flappy_dqn python=3.11 -y

# Activate the environment
conda activate flappy_dqn
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> [!TIP]
> If you encounter build issues with standard `pygame` on newer Python releases, this project uses `pygame-ce` (Pygame Community Edition) in `requirements.txt`, which provides pre-compiled binary wheels for modern Python versions.

---

## 🚀 How to Run

### 🎮 1. Manual Play Mode (Human Player)
Play the game yourself using the keyboard (press `SPACE` to flap):
```bash
python main.py
```

### 🏋️ 2. Train the RL Agent
Train the DQN agent from scratch using the hyperparameter profile `flappybirdv0` defined in `parameters.yaml`:
```bash
python agent.py flappybirdv0 --train
```
- **Checkpoints**: The best-performing model weights will be saved automatically to `runs/flappybirdv0.pt`.
- **Logs**: Step-by-step reward logs and training metrics will be saved to `runs/flappybirdv0.log`.

### 🤖 3. Test / Evaluate the Trained Agent
Watch the trained agent play the game in real-time (`render_mode="human"`):
```bash
python agent.py flappybirdv0
```

---

## ⚙️ Hyperparameter Configuration (`parameters.yaml`)

You can tune training parameters inside `parameters.yaml`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `env_name` | `FlappyBird-v0` | Gymnasium environment identifier |
| `alpha` | `0.001` | Learning rate for the Adam optimizer |
| `gamma` | `0.99` | Discount factor for future rewards ($\gamma$) |
| `epislon_init` | `1.0` | Initial exploration rate (100% random actions at start) |
| `epislon_min` | `0.05` | Minimum exploration rate (exploitation floor) |
| `epislon_decay` | `0.995` | Multiplicative decay factor applied to $\epsilon$ per episode |
| `replay_memory_size` | `100000` | Maximum capacity of the Experience Replay buffer |
| `minibatch_size` | `32` | Number of transition samples drawn per optimization step |
| `network_sync_rate` | `10` | Steps interval to sync Policy Network weights into Target Network |
| `reward_threshold` | `1000` | Episode reward threshold to consider an episode successful |

---

## 🔬 How the DQN Algorithm Works

1. **State Observation**: The continuous state vector from `FlappyBird-v0` (bird vertical position, velocity, distance to next pipes, pipe heights) is passed into the network.
2. **Action Selection**: An $\epsilon$-greedy policy balances exploration (random flap) and exploitation (action with $\max Q(s, a)$ computed by the Policy DQN).
3. **Experience Replay**: Transitions $(s, a, s', r, d)$ are stored in a cyclic memory buffer to break temporal correlation across training batches.
4. **Target Network Synchronization**: A separate target network stabilizes training by providing consistent Q-value targets while updating policy weights via Mean Squared Error (MSE) loss:
   $$y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')$$

---

## ❓ Troubleshooting & FAQs

### `ModuleNotFoundError: No module named 'setuptools._distutils.msvccompiler'`
- **Cause**: Standard `pygame` does not have prebuilt wheels for your Python version (such as Python 3.14), forcing pip to attempt building from source with older setuptools distutils dependencies.
- **Solution**: Install `pygame-ce` which provides prebuilt wheels:
  ```bash
  pip install pygame-ce
  ```
  Or recreate the virtual environment with Python 3.11 or 3.12:
  ```bash
  py -3.11 -m venv venv
  .\venv\Scripts\activate
  pip install -r requirements.txt
  ```
