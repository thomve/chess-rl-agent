import chess
import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

WIDTH, HEIGHT = 512, 512
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION

# Colors
WHITE = (240, 217, 181)
BROWN = (181, 136, 99)

# Load assets
IMAGES = {}

def load_images():
    pieces = ['wP', 'wR', 'wN', 'wB', 'wQ', 'wK',
              'bP', 'bR', 'bN', 'bB', 'bQ', 'bK']
    for piece in pieces:
        IMAGES[piece] = pygame.transform.scale(
            pygame.image.load(f"assets/{piece}.svg"), (SQ_SIZE, SQ_SIZE)
        )

def draw_board(screen, board):
    colors = [WHITE, BROWN]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[(r + c) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            col = chess.square_file(square)
            row = 7 - chess.square_rank(square)
            print(IMAGES)
            img = IMAGES[piece.color and 'w' or 'b' + piece.symbol().upper()]
            screen.blit(img, pygame.Rect(col*SQ_SIZE, row*SQ_SIZE, SQ_SIZE, SQ_SIZE))

# Basic neural net
class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(64*12, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.fc(x)

def board_to_tensor(board):
    tensor = np.zeros((12, 64), dtype=np.float32)
    piece_map = board.piece_map()
    for i in piece_map:
        p = piece_map[i]
        index = 'PNBRQKpnbrqk'.index(p.symbol())
        tensor[index][i] = 1
    return torch.tensor(tensor.flatten())

# Agent
class ChessAgent:
    def __init__(self, lr=0.001):
        self.model = QNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def choose_move(self, board):
        legal_moves = list(board.legal_moves)
        best_score = -float('inf')
        best_move = random.choice(legal_moves)
        for move in legal_moves:
            board.push(move)
            with torch.no_grad():
                x = board_to_tensor(board)
                score = self.model(x)
                if score > best_score:
                    best_score = score
                    best_move = move
            board.pop()
        return best_move

    def train(self, board, reward):
        x = board_to_tensor(board)
        target = torch.tensor([[reward]], dtype=torch.float32)
        pred = self.model(x)
        loss = self.loss_fn(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    screen.fill(pygame.Color("white"))
    load_images()
    board = chess.Board()

    agent_white = ChessAgent()
    agent_black = ChessAgent()

    running = True
    game_over = False

    while running:
        if board.is_game_over():
            print("Game over:", board.result())
            reward = 1 if board.result() == "1-0" else -1 if board.result() == "0-1" else 0
            agent_white.train(board, reward)
            agent_black.train(board, -reward)
            board.reset()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        draw_board(screen, board)
        pygame.display.flip()
        clock.tick(5)

        if not board.is_game_over():
            move = agent_white.choose_move(board) if board.turn == chess.WHITE else agent_black.choose_move(board)
            board.push(move)

    pygame.quit()

if __name__ == "__main__":
    main()
