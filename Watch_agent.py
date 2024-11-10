from PushBattle import Game
from random_agent import RandomAgent
from time import sleep
import agent
import matplotlib.pyplot as plt
import play_game as pg

def play_game(agent1, agent2):
    game = Game()
    game.BOARD_SIZE = 8
    game.NUM_PIECES = 8

    NUM_PIECES = 8
    EMPTY = 0
    #Starts off by swtiching to P1
    game.current_player = -1

    while True:
        game.display_board()
        print()
        game.current_player = game.current_player * -1

        current_pieces = game.p1_pieces if game.current_player == 1 else game.p2_pieces

        agent = agent1 if game.current_player == 1 else agent2
        move = agent.get_best_move(game)

        if current_pieces < NUM_PIECES:
            game.place_checker(move[0], move[1])
        else:
            game.move_checker(move[0], move[1], move[2], move[3])

        game.turn_count += 1

        winner = game.check_winner()
        if winner != EMPTY:
            game.display_board()
            print(f"Player {winner} wins!")
            return winner

        sleep(1)


def best_of_50(agent1, agent2):
    agent1_wins = 0
    agent2_wins = 0
    for _ in range(50):
        winner = pg.play_game(agent1, agent2, training=False)
        if winner == 1:
            agent1_wins += 1
        else:
            agent2_wins += 1

    return agent1_wins, agent2_wins


def tournament(agent1, agent2):
    agent1_white_wins, agent2_black_wins = best_of_50(agent1, agent2)
    agent2_white_wins, agent1_black_wins = best_of_50(agent2, agent1)

    # Calculate total wins
    agent1_total = agent1_white_wins + agent1_black_wins
    agent2_total = agent2_white_wins + agent2_black_wins

    # Set up the bar positions
    labels = ['White Wins', 'Black Wins', 'Total Wins']
    agent1_scores = [agent1_white_wins, agent1_black_wins, agent1_total]
    agent2_scores = [agent2_white_wins, agent2_black_wins, agent2_total]
    
    x = range(len(labels))
    width = 0.35

    # Create bars
    plt.bar([i - width/2 for i in x], agent1_scores, width, label='Agent 1')
    plt.bar([i + width/2 for i in x], agent2_scores, width, label='Agent 2')

    # Customize the plot
    plt.ylabel('Number of Wins')
    plt.title('Tournament Results')
    plt.xticks(x, labels)
    plt.legend()

    # Show the plot
    plt.show()

    return agent1_total, agent2_total


if __name__ == "__main__":
    Bot = agent.agent()
    Bot.load_model("saved_models/best_model.pt")

    tournament(Bot, RandomAgent())

    play_game(Bot, RandomAgent())
