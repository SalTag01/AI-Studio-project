import math
import random
import matplotlib.pyplot as plt
import numpy as np
import chess
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Converts a python-chess board into a (12, 8, 8) PyTorch FloatTensor."""
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)

        offset = 0 if piece.color == chess.WHITE else 6
        piece_idx = PIECE_TO_INDEX[piece.piece_type] + offset

        tensor[piece_idx, rank, file] = 1.0

    return torch.tensor(tensor)


def get_reward(
    board: chess.Board,
    is_game_over: bool,
    agent_color: chess.Color = chess.WHITE,
) -> float:
    """Calculates reward structure:

    - Win: +1.0 | Loss: -1.0 | Draw: 0.0
    - Intermediate steps: Material differential balance scaled down.
    """
    if is_game_over:
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            return 0.0  # Draw
        return 1.0 if outcome.winner == agent_color else -1.0

    # Intermediate Material Reward Heuristic
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    white_score = 0
    black_score = 0

    for piece in board.piece_map().values():
        val = piece_values.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            white_score += val
        else:
            black_score += val

    diff = (
        (white_score - black_score)
        if agent_color == chess.WHITE
        else (black_score - white_score)
    )
    return diff * 0.01  # Small step reward shaping



#deep q model 
class ChessDQN(nn.Module):
    """Convolutional Neural Network tailored to extract spatial board features and evaluate

    board state positions (Scalar Q-value per board state).
    """

    def __init__(self):
        super(ChessDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Evaluates position quality
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)



class ReplayBuffer:

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, reward, next_state, done):
        self.buffer.append((state, reward, next_state, done))

    def sample(self, batch_size: int):
        state, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.stack(state),
            torch.tensor(reward, dtype=torch.float32),
            torch.stack(next_state),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = ChessDQN().to(device)
target_net = ChessDQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
memory = ReplayBuffer(capacity=5000)

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 400
TARGET_UPDATE = 10
NUM_EPISODES = 300


def select_move(
    board: chess.Board, policy_net: nn.Module, epsilon: float
) -> chess.Move:
    """Epsilon-greedy legal action selection strategy."""
    legal_moves = list(board.legal_moves)

    # Exploration
    if random.random() < epsilon:
        return random.choice(legal_moves)

    # Exploitation: Pick move yielding highest expected Q-value
    best_move = legal_moves[0]
    best_val = -float("inf")

    for move in legal_moves:
        board.push(move)
        state_tensor = board_to_tensor(board).unsqueeze(0).to(device)
        with torch.no_grad():
            q_val = policy_net(state_tensor).item()
        board.pop()

        if q_val > best_val:
            best_val = q_val
            best_move = move

    return best_move


#training
episode_rewards = []
win_history = []

for episode in range(NUM_EPISODES):
    board = chess.Board()
    total_reward = 0.0
    steps = 0
    done = False

    # Calculate decaying Epsilon
    epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * episode / EPS_DECAY)

    while not board.is_game_over() and steps < 100:  # Max 100 moves per game
        current_state = board_to_tensor(board)

        # Agent Turn (White)
        move = select_move(board, policy_net, epsilon)
        board.push(move)
        steps += 1

        is_over = board.is_game_over() or steps >= 100
        reward = get_reward(board, is_over, agent_color=chess.WHITE)
        next_state = board_to_tensor(board)

        memory.push(current_state, reward, next_state, is_over)
        total_reward += reward

        if is_over:
            break

        # Environment/Opponent Turn (Black plays randomly)
        opponent_moves = list(board.legal_moves)
        board.push(random.choice(opponent_moves))

    episode_rewards.append(total_reward)

    # Outcome tracking
    outcome = board.outcome()
    if outcome and outcome.winner == chess.WHITE:
        win_history.append(1)
    else:
        win_history.append(0)

    # Deep Q-Network Optimization Step
    if len(memory) >= BATCH_SIZE:
        states, rewards, next_states, dones = memory.sample(BATCH_SIZE)
        states, rewards = states.to(device), rewards.to(device)
        next_states, dones = next_states.to(device), dones.to(device)

        # Current Q value estimates
        q_values = policy_net(states).squeeze(1)

        # Target Q value computation (Bellman equation)
        with torch.no_grad():
            max_next_q = target_net(next_states).squeeze(1)
            target_q_values = rewards + (1 - dones) * GAMMA * max_next_q

        loss = nn.MSELoss()(q_values, target_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update Target Network periodically
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if (episode + 1) % 20 == 0:
        avg_r = np.mean(episode_rewards[-20:])
        win_rate = np.mean(win_history[-20:]) * 100
        print(
            f"Episode {episode+1}/{NUM_EPISODES} | Avg Reward (last 20): {avg_r:.3f} | Win Rate: {win_rate:.1f}% | Epsilon: {epsilon:.2f}"
        )

#performance Visualization
plt.figure(figsize=(12, 5))

# Average Reward Plot
plt.subplot(1, 2, 1)
window = 20
smoothed_rewards = np.convolve(
    episode_rewards, np.ones(window) / window, mode="valid"
)
plt.plot(smoothed_rewards, color="#2b5c8f", linewidth=2)
plt.title("DQN Learning Curve (Moving Average Reward)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True, linestyle="--", alpha=0.6)

# Win Rate Plot
plt.subplot(1, 2, 2)
win_rates = [
    np.mean(win_history[max(0, i - window) : i + 1])
    for i in range(len(win_history))
]
plt.plot(win_rates, color="#2e7d32", linewidth=2)
plt.title("Agent Win Rate Progress (Rolling 20-Game Window)")
plt.xlabel("Episode")
plt.ylabel("Win Rate")
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()