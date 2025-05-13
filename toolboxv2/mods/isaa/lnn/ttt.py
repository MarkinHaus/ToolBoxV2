import numpy as np

from test import NeuralSystem


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        return self.board.flatten()

    def board_to_string(self):
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        return '\n---------\n'.join(['|'.join([symbols[cell] for cell in row]) for row in self.board])

    def make_move(self, action, watch=False):
        row, col = action // 3, action % 3
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player

            if watch:
                player_symbol = 'X' if self.current_player == 1 else 'O'
                print(
                    f"Spieler {self.current_player} ({player_symbol}) setzt auf Position {action} (Reihe {row}, Spalte {col})")
                print(self.board_to_string())
                print("-----------------")

            self.current_player = 3 - self.current_player  # Switch player (1 -> 2, 2 -> 1)
            return True

        if watch:
            print(f"Ungültiger Zug! Position {action} ist bereits besetzt.")

        return False

    def check_winner(self):
        # Check rows, columns, and diagonals
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                return self.board[i, 0] if self.board[i, 0] != 0 else self.board[0, i]
        if abs(sum(np.diag(self.board))) == 3 or abs(sum(np.diag(np.fliplr(self.board)))) == 3:
            return self.board[1, 1]
        return 0  # No winner

    def is_game_over(self):
        return self.check_winner() != 0 or np.all(self.board != 0)

    def get_valid_moves(self):
        return [i for i in range(9) if self.board[i // 3, i % 3] == 0]


import torch


class TicTacToeAgent:
    def __init__(self, neural_system, epsilon=0.1):
        self.neural_system = neural_system
        self.epsilon = epsilon

    def get_action(self, state, valid_moves):
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_moves)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.neural_system(state_tensor)

        valid_q_values = q_values[0, valid_moves]
        return valid_moves[torch.argmax(valid_q_values).item()]

    def train(self, state, action, next_state, reward, done):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        torch.tensor([action], dtype=torch.long)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        done_tensor = torch.tensor([done], dtype=torch.float32)

        with torch.no_grad():
            current_q = self.neural_system(state_tensor)[0]
            next_q = self.neural_system(next_state_tensor)[0].max().item()

        target_q = current_q.clone()
        target_q[action] = reward_tensor + (1 - done_tensor) * 0.99 * next_q

        _, loss = self.neural_system.train_step(state_tensor, None, target_q.unsqueeze(0))
        return loss


def train_agents(num_episodes=10000):
    input_size = 9
    hidden_sizes = [64, 64]
    output_size = 9
    liquid_state_hidden_size = 32
    liquid_state_num_layers = 2

    neural_system1 = NeuralSystem(input_size, hidden_sizes, output_size,
                                 liquid_state_hidden_size, liquid_state_num_layers)
    # neural_system2 = NeuralSystem(input_size, hidden_sizes, output_size,
    #                              liquid_state_hidden_size, liquid_state_num_layers)

    agent1 = TicTacToeAgent(neural_system1)
    agent2 = TicTacToeAgent(neural_system1)

    env = TicTacToe()

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_loss = 0
        moves = 0

        while not done:
            for agent in [agent1, agent2]:
                valid_moves = env.get_valid_moves()
                action = agent.get_action(state, valid_moves)
                env.make_move(action)
                next_state = env.get_state()
                winner = env.check_winner()
                done = env.is_game_over()

                if winner == 1:
                    reward = 1 if agent == agent1 else -1
                elif winner == 2:
                    reward = 1 if agent == agent2 else -1
                elif done:
                    reward = 0
                else:
                    reward = 0

                loss = agent.train(state, action, next_state, reward, done)
                total_loss += loss
                moves += 1
                state = next_state

                if done:
                    break

        if episode % 1000 == 0:
            print(
                f"Episode {episode}, Average Loss: {total_loss / moves:.4f}")  # 18 ist die maximale Anzahl von Zügen in einem Spiel

    return agent1, agent2


def evaluate_agents(agent1, agent2, num_games=10):
    env = TicTacToe()
    wins = {1: 0, 2: 0, 0: 0}

    for _ in range(num_games):
        state = env.reset()
        done = False

        while not done:
            for agent in [agent1, agent2]:
                valid_moves = env.get_valid_moves()
                action = agent.get_action(state, valid_moves)
                env.make_move(action, True)
                state = env.get_state()
                winner = env.check_winner()
                done = env.is_game_over()

                if done:
                    wins[winner] += 1
                    break

    print(f"Agent 1 wins: {wins[1]}")
    print(f"Agent 2 wins: {wins[2]}")
    print(f"Draws: {wins[0]}")


def play_against_ai(agent, human_player=1):
    env = TicTacToe()
    ai_player = 3 - human_player  # Wenn human_player 1 ist, ist ai_player 2 und umgekehrt

    while True:
        state = env.reset()
        done = False

        print("Neues Spiel beginnt!")
        print(board_to_string(state.reshape(3, 3)))
        print("")
        while not done:
            current_player = env.current_player

            if current_player == human_player:
                while True:
                    try:
                        action = int(input(f"Spieler {human_player}, gib deine Aktion ein (0-8): "))
                        if action in env.get_valid_moves():
                            break
                        else:
                            print("Ungültiger Zug. Bitte versuche es erneut.")
                    except ValueError:
                        print("Bitte gib eine Zahl zwischen 0 und 8 ein.")
            else:
                valid_moves = env.get_valid_moves()
                action = agent.get_action(state, valid_moves)

            env.make_move(action)
            state = env.get_state()
            print(board_to_string(state.reshape(3, 3)))
            print("")
            winner = env.check_winner()
            done = env.is_game_over()

            if done:
                if winner == human_player:
                    print("Glückwunsch! Du hast gewonnen!")
                elif winner == ai_player:
                    print("Die KI hat gewonnen!")
                else:
                    print("Unentschieden!")

        play_again = input("Möchtest du noch einmal spielen? (j/n): ").lower()
        if play_again != 'j':
            break

    print("Danke fürs Spielen!")


def board_to_string(board):
    symbols = {0: ' ', 1: 'X', 2: 'O'}
    return '\n---------\n'.join(['|'.join([symbols[cell] for cell in row]) for row in board])


if __name__ == "__main__":
    # Training starten
    agent1, agent2 = train_agents(4000)

    # Evaluierung starten
    evaluate_agents(agent1, agent2)
    play_against_ai(agent1, human_player=2)
    play_against_ai(agent2, human_player=2)
