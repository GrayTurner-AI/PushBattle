from PushBattle import Game

def play_game(player1, player2, training=False):
    game = Game()
    game.BOARD_SIZE = 8
    game.NUM_PIECES = 8

    NUM_PIECES = 8
    EMPTY = 0
    #Starts off by swtiching to P1
    game.current_player = -1

    while True:
        game.current_player = game.current_player * -1

        current_pieces = game.p1_pieces if game.current_player == 1 else game.p2_pieces

        agent = player1 if game.current_player == 1 else player2

        # Choose move based on whether we're training or evaluating
        if training:
            move = agent.get_training_move(game)
        else:
            move = agent.get_best_move(game)

        if current_pieces < NUM_PIECES:
            game.place_checker(move[0], move[1])
        else:
            game.move_checker(move[0], move[1], move[2], move[3])

        game.turn_count += 1

        winner = game.check_winner()
        if winner != EMPTY:
            return winner 
        


