"""
Reinforcement Learning Agent using Proximal Policy Optimization (PPO)
For adaptive scam honeypot engagement strategy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple
from collections import deque

class PolicyNetwork(nn.Module):
    """
    Policy network for action selection
    """
    def __init__(self, state_dim=256, action_dim=10, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        logits = self.network(state)
        return F.softmax(logits, dim=-1)

class ValueNetwork(nn.Module):
    """
    Value network for state value estimation
    """
    def __init__(self, state_dim=256, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)

class PPOHoneypotAgent:
    """
    PPO Agent for optimal honeypot engagement
    
    State: conversation history, extracted intel, scammer profile
    Actions: response strategy (question type, compliance level, detail level)
    Reward: +1 per intel extracted, +5 for critical intel, -10 if detected
    """
    def __init__(self, state_dim=256, action_dim=10, lr=3e-4, gamma=0.99, epsilon=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value = ValueNetwork(state_dim).to(self.device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon  # PPO clip parameter
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        
        # Action space definition
        self.action_space = {
            0: {"type": "feign_ignorance", "compliance": 0.2, "detail": "low"},  # "I don't understand"
            1: {"type": "provide_fake_info", "compliance": 1.0, "detail": "high"}, # "Here is my OTP"
            2: {"type": "express_fear", "compliance": 0.5, "detail": "medium"}, # "Will police come?"
            3: {"type": "ask_clarification", "compliance": 0.3, "detail": "medium"},
            4: {"type": "stall_time", "compliance": 0.4, "detail": "low"},
            5: {"type": "express_doubt", "compliance": 0.1, "detail": "high"},
            6: {"type": "partial_compliance", "compliance": 0.6, "detail": "medium"},
            7: {"type": "ask_verification", "compliance": 0.2, "detail": "high"},
            8: {"type": "show_urgency", "compliance": 0.8, "detail": "low"},
            9: {"type": "request_help", "compliance": 0.5, "detail": "high"}
        }
    
    def encode_state(self, session_data: Dict) -> np.ndarray:
        """
        Encode conversation state into fixed-size vector
        
        Args:
            session_data: {
                'turn_count': int,
                'intel_count': int,
                'critical_intel_count': int,
                'scam_confidence': float,
                'urgency_score': float,
                'emotion': str,
                'persona': str,
                'last_message_length': int,
                'manipulation_tactics': list
            }
        """
        state = np.zeros(256, dtype=np.float32)
        
        # Basic features
        state[0] = session_data.get('turn_count', 0) / 10.0  # Normalize
        state[1] = session_data.get('intel_count', 0) / 5.0
        state[2] = session_data.get('critical_intel_count', 0) / 3.0
        state[3] = session_data.get('scam_confidence', 0.0)
        state[4] = session_data.get('urgency_score', 0.0)
        
        # Emotion encoding (one-hot)
        emotions = ['confused', 'worried', 'scared', 'curious', 'skeptical', 'compliant']
        emotion = session_data.get('emotion', 'confused')
        if emotion in emotions:
            state[5 + emotions.index(emotion)] = 1.0
        
        # Persona encoding
        personas = ['elderly_confused', 'tech_unsavvy', 'concerned_parent', 'busy_professional']
        persona = session_data.get('persona', 'elderly_confused')
        if persona in personas:
            state[11 + personas.index(persona)] = 1.0
        
        # Manipulation tactics (multi-hot)
        tactics = ['urgency', 'threat', 'authority_claim', 'reward_lure', 'trust_building']
        detected_tactics = session_data.get('manipulation_tactics', [])
        for i, tactic in enumerate(tactics):
            if tactic in detected_tactics:
                state[15 + i] = 1.0
        
        return state
    
    def select_action(self, state: np.ndarray, training=False) -> Tuple[int, Dict]:
        """
        Select action using policy network
        
        Returns:
            action_id: int
            action_params: dict
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy(state_t)
            value = self.value(state_t)
        
        # Sample action
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Store for training
        if training:
            self.states.append(state)
            self.actions.append(action.item())
            self.log_probs.append(log_prob.item())
            self.values.append(value.item())
        
        action_id = action.item()
        action_params = self.action_space[action_id]
        
        return action_id, action_params
    
    def compute_reward(self, session_update: Dict) -> float:
        """
        Compute reward based on session update
        
        Args:
            session_update: {
                'new_intel_count': int,
                'new_critical_intel': int,
                'detected': bool,
                'conversation_ended': bool,
                'scammer_replied': bool
            }
        """
        reward = 0.0
        
        # 1. Scammer replied (Success: time wasted)
        if session_update.get('scammer_replied', False):
            reward += 2.0
        
        # 2. Scammer shared bank/UPI (Success: intel gathered)
        # We assume 'new_critical_intel' tracks this
        reward += session_update.get('new_critical_intel', 0) * 5.0
        
        # 3. Scammer stopped replying/detected (Failure: game over)
        if session_update.get('conversation_ended', False) and not session_update.get('scammer_replied', False):
             # If conversation ended without a reply (dropped) or detected
             reward -= 10.0
        elif session_update.get('detected', False):
             reward -= 10.0

        return reward
    
    def update(self, final_reward: float):
        """
        PPO update step
        """
        if len(self.states) == 0:
            return
        
        # Convert to tensors
        states_t = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_t = torch.LongTensor(self.actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(self.log_probs).to(self.device)
        
        # Compute returns (discounted rewards)
        returns = []
        R = final_reward
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns_t = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        
        # PPO update
        for _ in range(4):  # Multiple epochs
            # Get current policy and value
            action_probs = self.policy(states_t)
            values = self.value(states_t).squeeze()
            
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()
            
            # Compute advantages
            advantages = returns_t - values.detach()
            
            # Policy loss (PPO clipped objective)
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            
            # Value loss
            value_loss = F.mse_loss(values, returns_t)
            
            # Update
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        # Clear buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
    
    def save(self, path: str):
        """Save model weights"""
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
