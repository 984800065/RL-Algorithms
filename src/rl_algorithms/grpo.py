import gymnasium as gym
import torch
from torch import nn
from tqdm import tqdm
import numpy as np


G = 32
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.actor(x)


def get_group_process_advantages(group_rewards: torch.Tensor, group_masks: torch.Tensor) -> torch.Tensor:
    r_mean = group_rewards.mean()
    r_std = group_rewards.std()
    advantages_uncumsumed = (group_rewards - r_mean) / (r_std + 1e-8) * group_masks
    advantages = torch.flip(torch.cumsum(torch.flip(advantages_uncumsumed, dims=(1,)), dim=1), dims=(1,))
    return advantages

def grpo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    group_states: torch.Tensor,
    group_actions: torch.Tensor,
    old_group_log_probs: torch.Tensor,
    group_advantages: torch.Tensor,
    group_masks: torch.Tensor,
    update_epochs: int = 4,
    minibatch_size: int = 512,
    beta: float = 0.01,
    entropy_coef: float = 0.01,
    clip_epsilon: float = 0.2
):
    advantages_mean = group_advantages.sum() / group_masks.sum()
    advantages_std = torch.sqrt(((group_advantages - advantages_mean) * group_masks).pow(2).sum() / group_masks.sum())
    group_advantages = (group_advantages - advantages_mean) / (advantages_std + 1e-8)

    model.train()
    G = group_states.shape[0]
    for epoch in range(update_epochs):
        ref_group_log_probs = old_group_log_probs.clone()
        for g in range(G):
            for begin_idx in range(0, int(group_masks[g].sum()), minibatch_size):
                end_idx = begin_idx + minibatch_size
                batch_idx = slice(begin_idx, end_idx)
                
                probs: torch.Tensor = model(group_states[g, batch_idx])
                dis = torch.distributions.Categorical(probs)
                log_probs = dis.log_prob(group_actions[g, batch_idx])
                entropy = dis.entropy().mean()
                ratio = torch.exp(log_probs - old_group_log_probs[g, batch_idx])

                clip_loss = -torch.min(
                    ratio * group_advantages[g, batch_idx], 
                    torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * group_advantages[g, batch_idx]
                ).mean()
                ref_ratio = torch.exp(ref_group_log_probs[g, batch_idx] - log_probs)
                KL_loss = beta * (ref_ratio - torch.log(ref_ratio) - 1).mean()
                entropy_bonus = -entropy_coef * entropy
                loss = clip_loss + KL_loss + entropy_bonus
                loss /= G

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
def train(
    vec_env: gym.Env, 
    model: ActorCritic, 
    state_dim: int,
    actor_lr: float = 1e-4, 
    max_episodes: int = 100, 
    G: int = G, 
    max_steps: int = 512
):
    optimizer = torch.optim.AdamW([
        {"params": model.actor.parameters(), "lr": actor_lr},
    ])

    pbar = tqdm(range(max_episodes))
    for episode in pbar:
        # 采样G组数据
        model.eval()
        group_states = torch.zeros((G, max_steps, state_dim), dtype=torch.float, device=device)
        group_actions = torch.zeros((G, max_steps), dtype=torch.long, device=device)
        group_log_probs = torch.zeros((G, max_steps), dtype=torch.float, device=device)
        group_rewards = torch.zeros((G, max_steps), dtype=torch.float, device=device)
        group_masks = torch.zeros((G, max_steps), dtype=torch.float, device=device)
        finish_judger = np.array([False] * G)
        with torch.no_grad():
            states, _ = vec_env.reset()
            for step in range(max_steps):
                states_tensor = torch.tensor(states, dtype=torch.float).to(device)
                probs: torch.Tensor = model(states_tensor)
                dis = torch.distributions.Categorical(probs)
                actions = dis.sample()
                next_states, rewards, dones, _, _ = vec_env.step(actions.cpu().numpy().astype(np.int32))

                group_states[:, step] = states_tensor
                group_actions[:, step] = actions
                group_log_probs[:, step] = dis.log_prob(actions)
                group_rewards[:, step] = torch.tensor(rewards, dtype=torch.float, device=device)
                group_masks[:, step] = torch.tensor(1 - dones, dtype=torch.float, device=device)
                # print(dones)
                # print(rewards)

                states = next_states
                finish_judger = finish_judger | dones
                # print(finish_judger)
                if all(finish_judger):
                    break

        zero_after = (1 - group_masks).cumsum(dim=1) > 0
        group_rewards[zero_after] = 0
        group_masks[zero_after] = 0
        group_advantages = get_group_process_advantages(group_rewards, group_masks)

        grpo_update(model, optimizer, group_states, group_actions, group_log_probs, group_advantages, group_masks)

        # print(f"Episode {episode + 1}, Reward: {avg_reward}")
        pbar.set_postfix(group_reward=group_rewards.sum() / G)


if __name__ == "__main__":
    vec_env = gym.make_vec("CartPole-v1", num_envs=G)
    state_dim, action_dim = vec_env.observation_space.shape[1], vec_env.action_space[0].n
    model = ActorCritic(state_dim, action_dim).to(device)
    train(vec_env, model, state_dim)