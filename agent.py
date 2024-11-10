import model
import play_game
from collections import deque
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import TensorBoardLogger
import os
from datetime import datetime



class agent():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.ActorCriticNet(num_channels=64, num_blocks=15)
        self.model = self.model.to(self.device)
        self.memory = deque(maxlen=50000)
        self.last_state = None
        self.training_iteration = 0
        
        # Create TensorBoard logger with unique directory for each model
        model_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join('logs', 'a2c_training')
        self.base_log_dir = log_dir
        self.model_id = model_id
        self.logger = None  # We'll create a new logger for each training step
        
        # Add save directory using same model_id
        self.save_dir = os.path.join('saved_models', model_id)
        os.makedirs(self.save_dir, exist_ok=True)

    def get_best_move(self, game):
        # Process board state
        state = self.process_board(game).to(self.device)
        
        # Update next_state for previous transition if exists
        if self.memory:
            self.memory[-1]['next_state'] = state
        
        # Get policy predictions from model
        policy_pred, _ = self.model(state.unsqueeze(0))
        
        # Get valid moves
        current_pieces = game.p1_pieces if game.current_player == 1 else game.p2_pieces
        valid_moves = self.get_possible_moves(game, current_pieces)
        
        # Make sure both tensors are 1D before multiplication
        policy_flat = policy_pred.reshape(-1)
        mask_flat = self.board_mask(game.board, valid_moves).to(self.device).reshape(-1)
        
        # Element-wise multiplication
        move_probs = policy_flat * mask_flat
        
        # Get the move with highest probability
        move_idx = torch.argmax(move_probs).item()
        
        # Map the index back to valid moves
        valid_indices = torch.nonzero(mask_flat).squeeze()
        move_list_idx = torch.where(valid_indices == move_idx)[0][0]
        move = valid_moves[move_list_idx]
        
        # Convert move tuple to tensor index before storing
        if len(move) == 2:  # Placement move
            move_idx = move[0] * game.board.shape[1] + move[1]
        else:  # Movement move
            move_idx = move[2] * game.board.shape[1] + move[3]
        
        # Store state and action for training
        transition = {
            'state': state,
            'action': torch.tensor(move_idx),
            'reward': None,
            'next_state': None
        }
        self.memory.append(transition)
        
        return move

    def get_training_move(self, game):
        # Process board state
        state = self.process_board(game).to(self.device)
        
        # Update next_state for previous transition if exists
        if self.memory:
            self.memory[-1]['next_state'] = state
        
        # Get policy predictions from model
        policy_pred, _ = self.model(state.unsqueeze(0))
        
        # Get valid moves
        current_pieces = game.p1_pieces if game.current_player == 1 else game.p2_pieces
        valid_moves = self.get_possible_moves(game, current_pieces)
        
        # Make sure both tensors are 1D before multiplication
        policy_flat = policy_pred.reshape(-1)
        mask_flat = self.board_mask(game.board, valid_moves).to(self.device).reshape(-1)
        
        # Element-wise multiplication
        move_probs = policy_flat * mask_flat
        
        # Sample from the probability distribution of legal moves
        move_idx = torch.multinomial(move_probs, num_samples=1).item()
        
        # Map the flattened index back to valid moves
        valid_indices = torch.nonzero(mask_flat).squeeze()
        move_list_idx = torch.where(valid_indices == move_idx)[0][0]
        move = valid_moves[move_list_idx]
        
        # Convert move tuple to tensor index before storing
        if len(move) == 2:  # Placement move
            move_idx = move[0] * game.board.shape[1] + move[1]
        else:  # Movement move
            move_idx = move[2] * game.board.shape[1] + move[3]
        
        # Store state and action for training
        transition = {
            'state': state,
            'action': torch.tensor(move_idx),
            'reward': None,
            'next_state': None
        }
        self.memory.append(transition)
        
        return move

    def process_board(self, game):
        if game.current_player == 1:
            board = game.board
        else:
            board = self.flip_board(game.board)

        return torch.FloatTensor(board).to(self.device).unsqueeze(0)
    

    def flip_board(self, board):
        """Flips the values of 1 and -1 in an 8x8 board while keeping 0s unchanged."""
        flipped_board = board.copy()
        flipped_board[board == 1] = -1
        flipped_board[board == -1] = 1
        return flipped_board
    

    def get_possible_moves(self, game, current_pieces):
        """Returns list of all possible moves in current state."""
        moves = []
        board = game.board
        
        if current_pieces < game.NUM_PIECES:
            # Get empty positions in one pass using numpy
            empty_positions = np.argwhere(board == 0)
            moves = [(pos[0], pos[1]) for pos in empty_positions]
        else:
            # Get player pieces and empty spaces in one pass
            player_positions = np.argwhere(board == game.current_player)
            empty_positions = np.argwhere(board == 0)
            
            # Create moves by combining player positions with empty positions
            for p_pos in player_positions:
                for e_pos in empty_positions:
                    moves.append((p_pos[0], p_pos[1], e_pos[0], e_pos[1]))
                    
        return moves

    def board_mask(self, board, valid_moves):
        """Creates a mask of valid moves by checking game rules"""
        mask = torch.zeros((board.shape[0]*board.shape[1], 1))
        for move in valid_moves:
            if len(move) == 2:  # Placement move
                # Convert 2D position to 1D index
                idx = move[0] * board.shape[1] + move[1]
                if board[move[0]][move[1]] == 0:  # Check if position is empty
                    mask[idx] = 1
            else:  # Movement move
                src_r, src_c, dst_r, dst_c = move
                # Convert destination 2D position to 1D index
                idx = dst_r * board.shape[1] + dst_c
                current_player = board[src_r][src_c]
                if (current_player != 0 and board[dst_r][dst_c] == 0):  
                    mask[idx] = 1
        return mask.to(self.device)

    def play_game(self, training=False):
        reward = play_game.play_game(self, self, training=training)
        # Update final state's next_state (terminal state)
        if self.memory:
            self.memory[-1]['next_state'] = self.memory[-1]['state']  # Terminal state transitions to itself
        
        # Alternate rewards for white/black since model plays both sides
        for i in range(len(self.memory)):
            self.memory[i]['reward'] = reward if i % 2 == 0 else -reward

        


    def collect_training_data(self, num_games=10):
        print(f"\nCollecting data from {num_games} self-play games (iteration {self.training_iteration})")
        for i in range(num_games):
            self.play_game(training=True)  # Set training flag to True
            if (i + 1) % 100 == 0:  # Progress update every 100 games
                print(f"Completed {i + 1}/{num_games} games")

        self.train()
        self.training_iteration += 1
        
        # Save model after training
        self.save_model()


    def train(self, epochs=100, batch_size=32, learning_rate=0.001):
        # Set precision for matmul
        torch.set_float32_matmul_precision('medium')
        
        # Create dataset from memory
        states = torch.stack([m['state'] for m in self.memory])
        actions = torch.stack([m['action'] for m in self.memory])
        rewards = torch.tensor([m['reward'] for m in self.memory], device=self.device).float()
        next_states = torch.stack([m['next_state'] for m in self.memory])

        # Calculate advantages with GAE (Generalized Advantage Estimation)
        gamma = 0.99
        lambda_ = 0.95
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        with torch.no_grad():
            _, values = self.model(states)
            _, next_values = self.model(next_states)
            values = values.squeeze()
            next_values = next_values.squeeze()
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = next_values[t + 1]
                
                delta = rewards[t] + gamma * next_value - values[t]
                gae = delta + gamma * lambda_ * gae
                advantages[t] = gae
                
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Create dataset
        dataset = TensorDataset(states, actions, rewards, advantages)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Create new logger for this training step
        step_log_dir = os.path.join(self.base_log_dir, self.model_id, f'step_{self.training_iteration}')
        self.logger = TensorBoardLogger(
            save_dir=step_log_dir,
            name='',  # Empty name to avoid extra subfolder
            version='',  # Empty version to avoid extra subfolder
            default_hp_metric=False,
            log_graph=False
        )

        # Create a new instance of the model for training
        training_model = model.ActorCriticNet(
            num_channels=64, 
            num_blocks=15,
            learning_rate=learning_rate,
            training_iteration=self.training_iteration
        ).to(self.device)
        
        # For first iteration, or if you want to start fresh
        if self.training_iteration == 0:
            # Initialize with fresh weights
            self.model = model.ActorCriticNet(
                num_channels=64,
                num_blocks=15,
                learning_rate=learning_rate,
                training_iteration=0
            ).to(self.device)
        
        # Copy the current model's state
        training_model.load_state_dict(self.model.state_dict())
        
        # Reset model buffers and gradients
        training_model.zero_grad()
        for param in training_model.parameters():
            param.grad = None
        
        # Setup trainer with minimal verbosity
        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=[
                EarlyStopping(
                    monitor='val_policy_loss',
                    min_delta=0.00,
                    patience=5,
                    verbose=False,
                    mode='min'
                ), 
                EarlyStopping(
                    monitor='val_value_loss',
                    min_delta=0.00,
                    patience=5,
                    verbose=False,
                    mode='min'
                )
            ],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            logger=self.logger,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            log_every_n_steps=1
        )

        # Run learning rate finder
        tuner = Tuner(trainer)
        try:
            lr_finder = tuner.lr_find(
                training_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                min_lr=1e-6,
                max_lr=1e-2,
                num_training=200,
                mode='exponential',
                early_stop_threshold=4.0
            )

            if lr_finder.suggestion() is not None:
                new_lr = lr_finder.suggestion()
                # Bound the learning rate
                new_lr = max(1e-5, min(1e-3, new_lr))
                print(f"Learning rate set to: {new_lr}")
            else:
                print(f"Using default learning rate: {learning_rate}")
                new_lr = learning_rate
                
            # Update the model's learning rate
            for param_group in training_model.configure_optimizers()['optimizer'].param_groups:
                param_group['lr'] = new_lr
                
        except Exception as e:
            print(f"Learning rate finder failed: {str(e)}")
            print(f"Using default learning rate: {learning_rate}")

        # Train the model
        trainer.fit(training_model, train_loader, val_loader)
        
        # Update the agent's model with the trained weights
        self.model.load_state_dict(training_model.state_dict())

    def save_model(self):
        """Save the current model state"""
        save_path = os.path.join(self.save_dir, f'model_iteration_{self.training_iteration}.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_iteration': self.training_iteration,
        }, save_path)

    def load_model(self, path):
        """Load a saved model state"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_iteration = checkpoint['training_iteration']
        self.model = self.model.to(self.device)

    def evaluate_against_previous(self, num_games=100):
        """
        Evaluates current model against previous best model in a series of games.
        Returns True if current model wins majority of games.
        """
        # Create a copy of the current model
        current_model = model.ActorCriticNet(num_channels=64, num_blocks=15).to(self.device)
        current_model.load_state_dict(self.model.state_dict())
        
        # Create an agent with the previous best model
        previous_agent = agent()
        best_model_path = os.path.join('saved_models', 'best_model.pt')
        if os.path.exists(best_model_path):
            previous_agent.load_model(best_model_path)
        else:
            # If no best model exists yet, current model becomes the best
            self.save_as_best()
            return True
        
        # Track wins
        current_model_wins = 0
        
        print(f"\nEvaluating against previous best model ({num_games} games)...")
        for i in range(num_games):
            # Alternate who plays first, using training=False for evaluation
            if i % 2 == 0:
                winner = play_game.play_game(self, previous_agent, training=False)
                if winner == 1:  # Current model wins
                    current_model_wins += 1
            else:
                winner = play_game.play_game(previous_agent, self, training=False)
                if winner == -1:  # Current model wins
                    current_model_wins += 1
                    
            if (i + 1) % 10 == 0:  # Progress update every 10 games
                print(f"Completed {i + 1}/{num_games} games. Current model wins: {current_model_wins}")
        
        win_rate = current_model_wins / num_games
        print(f"\nEvaluation complete. Current model win rate: {win_rate:.2%}")
        
        # If current model wins majority, save it as the best model
        if win_rate > 0.52:  # Changed from 0.55 to 0.52
            self.save_as_best()
            return True
        return False

    def save_as_best(self):
        # Save the current model as the best model
        best_model_path = os.path.join('saved_models', 'best_model.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_iteration': self.training_iteration,
        }, best_model_path)


class TrainAgent():
    def __init__(self, agent, epochs, batch_size, learning_rate, num_games, num_updates):
        self.agent = agent
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_games = num_games
        self.num_updates = num_updates
        self.train()

    def train(self):
        for i in range(self.num_updates):
            self.agent.collect_training_data(self.num_games)
            print(f"\nEvaluating model after update {i+1}/{self.num_updates}")
            self.agent.evaluate_against_previous()
            self.agent.memory.clear()


if __name__ == "__main__":
    A2C = agent()
    train_agent = TrainAgent(
        A2C,
        epochs=50,
        batch_size=64,
        learning_rate=0.001,
        num_games=750,
        num_updates=500
    )


        



