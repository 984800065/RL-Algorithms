import gymnasium as gym
import torch
from torch import nn


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared)


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, masks: torch.Tensor, gamma: float = 0.99, lam: float = 0.95):
    values_ext = torch.cat([values, torch.zeros(1, dtype=torch.float, device=device)])
    gae = torch.zeros(1, dtype=torch.float, device=device)
    returns = torch.zeros_like(rewards)
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values_ext[step + 1] * masks[step] - values_ext[step]
        gae = delta + gamma * lam * gae * masks[step]
        returns[step] = gae + values_ext[step]

    return returns

def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    update_epochs: int = 4,
    minibatch_size: int = 64,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    clip_epsilon: float = 0.2
):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    model.train()
    for epoch in range(update_epochs):
        for start_idx in range(0, len(states), minibatch_size):
            end_idx = start_idx + minibatch_size
            batch_idx = slice(start_idx, end_idx)
            
            probs: torch.Tensor
            values: torch.Tensor
            probs, values = model(states[batch_idx])
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions[batch_idx])
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs[batch_idx])
            clip_loss = -torch.min(
                ratio * advantages[batch_idx], 
                torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages[batch_idx]
            ).mean()  # 最大化期望收益
            critic_loss = value_coef * (returns[batch_idx] - values.squeeze()).pow(2).mean()  # 最小化价值预测与真实值差异
            entropy_bonus = -entropy_coef * entropy  # 最大化熵

            loss = clip_loss + critic_loss + entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train(
    env: gym.Env, 
    model: ActorCritic, 
    max_steps: int, 
    max_episodes: int, 
    shared_lr: float = 3e-4, 
    actor_lr: float = 3e-4, 
    critic_lr: float = 1e-3
):
    optimizer = torch.optim.AdamW([
        {"params": model.shared.parameters(), "lr": shared_lr},
        {"params": model.actor.parameters(), "lr": actor_lr},
        {"params": model.critic.parameters(), "lr": critic_lr}
    ])

    for episode in range(max_episodes):
        state, _ = env.reset()

        states, actions, rewards, values, log_probs, masks = [], [], [], [], [], []

        model.eval()
        with torch.no_grad():
            for step in range(max_steps):
                state_tensor = torch.tensor(state, dtype=torch.float).to(device)
                probs: torch.Tensor
                value: torch.Tensor
                probs, value = model(state_tensor)
                dis = torch.distributions.Categorical(probs)
                action = dis.sample()
                next_state, reward, done, _, _ = env.step(action.item())
                states.append(state_tensor)
                actions.append(action)
                rewards.append(reward)
                masks.append(1 - done)
                log_probs.append(dis.log_prob(action))
                values.append(value)

                state = next_state
                if done:
                    break

        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        masks = torch.tensor(masks, dtype=torch.float).to(device)
        log_probs = torch.stack(log_probs).to(device)
        values = torch.stack(values).to(device).squeeze()

        returns = compute_gae(rewards, values, masks)
        advantages = returns - values

        ppo_update(model, optimizer, states, actions, log_probs, advantages, returns)

        if (episode + 1) % 10 == 0:
            total_reward = sum(rewards)
            print(f"Episode {episode + 1}, Reward: {total_reward}")


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.n
    model = ActorCritic(state_dim, action_dim).to(device)
    train(env, model, 512, 100)
    # torch.save(model.state_dict(), "./model_parameters/ppo.pt")