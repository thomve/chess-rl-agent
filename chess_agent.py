import chess
import chess.engine
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return list(self.memory)
        else:
            return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

memory = ReplayMemory(capacity=10000)


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

board = chess.Board()

input_size = 8*8*6
hidden_size = 256
output_size = 4096  # 8x8x64 possible moves


model = DQN(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()


def board_to_input(board):
    state = np.zeros((8, 8, 6), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(i, j))
            if piece is not None:
                state[i, j, piece.piece_type - 1] = 1
    return state.flatten()

def select_action(state, epsilon):
    legal_moves = [move for move in board.legal_moves]
    if random.random() < epsilon:
        return random.choice(legal_moves)
    else:
        print("exploitation")
        with torch.no_grad():
            q_values = model(torch.tensor(state))
            legal_moves_index = [uci_to_index(i.uci()) for i in legal_moves]
            legal_q_values = [q_values[move] if move in legal_moves_index else float('-inf') for move in range(len(q_values))]
            return torch.argmax(torch.tensor(legal_q_values)).item()

def execute_action(action):
    move = chess.Move.from_uci(action.uci())
    board.push(move)


def index_to_uci(index):
    from_index = index // 64
    to_index = index % 64
    
    from_file = chr(from_index % 8 + ord('a'))
    from_rank = str(from_index // 8 + 1)
    
    to_file = chr(to_index % 8 + ord('a'))
    to_rank = str(to_index // 8 + 1)
    
    uci = from_file + from_rank + to_file + to_rank
    return uci


def uci_to_index(uci):
    from_square, to_square = uci[:2], uci[2:]
    
    from_file, from_rank = ord(from_square[0]) - ord('a'), int(from_square[1]) - 1
    to_file, to_rank = ord(to_square[0]) - ord('a'), int(to_square[1]) - 1
    
    from_index = from_rank * 8 + from_file
    to_index = to_rank * 8 + to_file
    
    return from_index * 64 + to_index


epsilon = 1.0
epsilon_decay = 0.995
gamma = 0.99
num_episodes = 1000
batch_size = 250

for episode in range(num_episodes):
    state = board_to_input(board)
    done = False
    total_reward = 0

    while not done:
        action = select_action(state, epsilon)
        if isinstance(action, int):
            action = chess.Move.from_uci(index_to_uci(action))
        execute_action(action)
        reward = 1 if board.is_checkmate() else 0  # Reward is 1 for checkmate, 0 otherwise
        total_reward += reward

        next_state = board_to_input(board)
        action_encoded = uci_to_index(action.uci())
        memory.push(state, action_encoded, next_state, reward)
        state = next_state

        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_batch = torch.tensor(batch.state, dtype=torch.float32)
        action_batch = torch.tensor(batch.action, dtype=torch.int64)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32)

        current_q_values = model(state_batch).gather(1, action_batch.view(-1, 1))

        # Compute max_a Q(s_{t+1}, a)
        next_max_q = model(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + (gamma * next_max_q)

        loss = loss_fn(current_q_values, target_q_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
            done = True

    epsilon *= epsilon_decay
    epsilon = max(0.1, epsilon)  # Ensure epsilon doesn't fall below 0.1

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}")


torch.save(model.state_dict(), "chess_agent.pth")
