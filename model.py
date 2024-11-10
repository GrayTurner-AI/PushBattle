import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

class ResidualBlock(pl.LightningModule):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class ActorCriticNet(pl.LightningModule):
    def __init__(self, num_channels=64, num_blocks=10, learning_rate=0.001, training_iteration=0):
        super().__init__()
        self.save_hyperparameters()
        
        # Shared feature extractor
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        # Shared residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_blocks)
        ])

        # Actor head (policy network)
        self.actor_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.actor_bn = nn.BatchNorm2d(32)
        self.actor_final = nn.Conv2d(32, 1, kernel_size=1)
        
        # Critic head (value network)
        self.critic_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.critic_bn = nn.BatchNorm2d(32)
        self.critic_flatten = nn.Flatten()
        self.critic_linear1 = nn.Linear(32 * 64, 256)
        self.critic_linear2 = nn.Linear(256, 1)

        # Add loss functions
        self.policy_criterion = nn.CrossEntropyLoss()  # For policy loss
        self.value_criterion = nn.MSELoss()  # For value loss
        self.learning_rate = learning_rate
        self.training_iteration = training_iteration

        # Initialize lists to store losses for the epoch
        self.train_policy_losses = []
        self.train_value_losses = []
        self.val_policy_losses = []
        self.val_value_losses = []

        self.automatic_optimization = True  # Ensure this is set

    def forward_shared(self, x):
        # Shared layers
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x = block(x)
        return x
        
    def forward_actor(self, x):
        # Actor head
        x = F.relu(self.actor_bn(self.actor_conv(x)))
        x = self.actor_final(x)
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)  # Flatten to (batch_size, 64)
        x = F.softmax(x, dim=1) + 1e-8
        return x
        
    def forward_critic(self, x):
        # Critic head
        x = F.relu(self.critic_bn(self.critic_conv(x)))
        x = self.critic_flatten(x)
        x = F.relu(self.critic_linear1(x))
        x = self.critic_linear2(x)
        return x
    
    def forward(self, x):
        shared_features = self.forward_shared(x)
        policy = self.forward_actor(shared_features)
        value = self.forward_critic(shared_features)
        return policy, value

    def _common_step(self, batch, batch_idx):
        states, actions, rewards, advantages = batch
        
        # Get model predictions
        policy_pred, value_pred = self(states)
        
        # Calculate policy loss using log probabilities and advantages
        log_probs = torch.log_softmax(policy_pred, dim=1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1))
        policy_loss = -(selected_log_probs * advantages.unsqueeze(1)).mean()
        
        # Calculate value loss
        value_pred = value_pred.squeeze()
        value_loss = self.value_criterion(value_pred, rewards)
        
        # Add entropy bonus with dynamic weight
        entropy = -(torch.softmax(policy_pred, dim=1) * log_probs).sum(dim=1).mean()
        entropy_weight = max(0.001, 0.1 * np.exp(-0.001 * self.training_iteration))
        
        # Combined loss
        loss = policy_loss + 0.5 * value_loss - (entropy_weight * entropy)
        
        return loss, policy_loss, value_loss

    def training_step(self, batch, batch_idx):
        # Get losses from common step
        loss, policy_loss, value_loss = self._common_step(batch, batch_idx)
        
        # Store losses for epoch end
        self.train_policy_losses.append(policy_loss)
        self.train_value_losses.append(value_loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        # Get losses from common step
        loss, policy_loss, value_loss = self._common_step(batch, batch_idx)
        
        # Store losses for epoch end
        self.val_policy_losses.append(policy_loss)
        self.val_value_losses.append(value_loss)
        
        return loss

    def on_train_epoch_end(self):
        # Calculate and log average losses for the epoch
        if self.train_policy_losses:
            avg_policy_loss = torch.stack(self.train_policy_losses).mean()
            avg_value_loss = torch.stack(self.train_value_losses).mean()
            
            self.log_dict({
                'train_policy_loss': avg_policy_loss,
                'train_value_loss': avg_value_loss,
            }, on_step=False, on_epoch=True, prog_bar=False)
            
            # Clear lists for next epoch
            self.train_policy_losses.clear()
            self.train_value_losses.clear()

    def on_validation_epoch_end(self):
        # Calculate and log average losses for the epoch
        if self.val_policy_losses:
            avg_policy_loss = torch.stack(self.val_policy_losses).mean()
            avg_value_loss = torch.stack(self.val_value_losses).mean()
            
            self.log_dict({
                'val_policy_loss': avg_policy_loss,
                'val_value_loss': avg_value_loss,
            }, on_step=False, on_epoch=True, prog_bar=False)
            
            # Clear lists for next epoch
            self.val_policy_losses.clear()
            self.val_value_losses.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_policy_loss',
            'gradient_clip_val': 1.0,
        }

    # Add these two required methods
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    # Add a method to set the dataloaders
    def set_dataloaders(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def policy_criterion(self, policy_pred, actions, advantages):
        log_probs = torch.log_softmax(policy_pred, dim=1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1))
        policy_loss = -(selected_log_probs * advantages.unsqueeze(1)).mean()
        return torch.abs(policy_loss)  # Take absolute value

    def reset_states(self):
        """Reset any stateful data"""
        self.zero_grad()
        for param in self.parameters():
            param.grad = None
            
    def on_train_start(self):
        """Called when training starts"""
        # Clear all loss lists
        self.train_policy_losses.clear()
        self.train_value_losses.clear()
        self.val_policy_losses.clear()
        self.val_value_losses.clear()

