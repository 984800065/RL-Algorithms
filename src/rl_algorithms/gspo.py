import os
import sys
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from torch import nn
from tqdm import tqdm
from loguru import logger


logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{file.path}</cyan>:<cyan>{line}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>"
)
logger.info("logger has been initialized")

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
    # advantages = (group_rewards - r_mean) / (r_std + 1e-8) * group_masks
    return advantages

def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str = "seq-mean-token-mean") -> torch.Tensor:
    if loss_agg_mode == "seq-mean-token-mean":
        seq_sum = (loss_mat * loss_mask).sum(dim=-1)
        seq_len = loss_mask.sum(dim=-1).clamp(min=1)
        seq_mean = seq_sum / seq_len
        return seq_mean.mean()
    elif loss_agg_mode == "token-mean":
        return (loss_mat * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1)
    else:
        raise ValueError(f"Invalid loss aggregation mode: {loss_agg_mode}")

def gspo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    group_states: torch.Tensor,
    group_actions: torch.Tensor,
    old_group_log_probs: torch.Tensor,
    group_advantages: torch.Tensor,
    group_masks: torch.Tensor,
    update_epochs: int = 4,
    minibatch_size: int = 512,
    beta: float = 1e-4,
    entropy_coef: float = 1e-4,
    clip_epsilon_low: float = 3e-4,
    clip_epsilon_high: float = 4e-4,
    kl_loss_on: bool = True,
    entropy_bonus_on: bool = True
):
    advantages_mean = group_advantages.sum() / group_masks.sum()
    advantages_std = torch.sqrt(((group_advantages - advantages_mean) * group_masks).pow(2).sum() / group_masks.sum())
    group_advantages = (group_advantages - advantages_mean) / (advantages_std + 1e-8)

    model.train()
    G = group_states.shape[0]
    for epoch in range(update_epochs):
        ref_group_log_probs = old_group_log_probs.clone().detach()
        group_log_probs = torch.zeros_like(old_group_log_probs, dtype=torch.float, device=device)
        group_entropy = torch.zeros_like(old_group_log_probs, dtype=torch.float, device=device)
        KL_loss = 0
        entropy_bonus = 0
        for g in range(G):
            for begin_idx in range(0, int(group_masks[g].sum()), minibatch_size):
                end_idx = begin_idx + minibatch_size
                batch_idx = slice(begin_idx, end_idx)

                probs: torch.Tensor = model(group_states[g, batch_idx])
                dis = torch.distributions.Categorical(probs)
                log_probs = dis.log_prob(group_actions[g, batch_idx])
                group_log_probs[g, batch_idx] = log_probs
                group_entropy[g, batch_idx] = dis.entropy()
        
        negative_approx_kl = group_log_probs - old_group_log_probs
        seq_lengths = torch.sum(group_masks, dim=-1).clamp(min=1)
        negative_approx_kl_seq = torch.sum(negative_approx_kl * group_masks, dim=-1) / seq_lengths

        log_seq_importance_ratio = group_log_probs - group_log_probs.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
        log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10)

        seq_importance_ratio = torch.exp(log_seq_importance_ratio)

        pg_losses1 = group_advantages * seq_importance_ratio
        pg_losses2 = group_advantages * torch.clamp(seq_importance_ratio, 1 - clip_epsilon_low, 1 + clip_epsilon_high)
        pg_losses = -torch.minimum(pg_losses1, pg_losses2)
        pg_loss = agg_loss(pg_losses, group_masks, "seq-mean-token-mean")

        log_ref_ratio = ref_group_log_probs - group_log_probs
        KL_losses = beta * (torch.exp(log_ref_ratio) - log_ref_ratio - 1)
        KL_loss = agg_loss(KL_losses, group_masks, "seq-mean-token-mean")

        entropy_bonuses = -entropy_coef * group_entropy
        entropy_bonus = agg_loss(entropy_bonuses, group_masks, "seq-mean-token-mean")

        # logger.info(f"pg_loss: {pg_loss}, KL_loss: {KL_loss}, entropy_bonus: {entropy_bonus}")

        loss = pg_loss
        if kl_loss_on:
            loss += KL_loss
        if entropy_bonus_on:
            loss += entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
def train(
    vec_env: gym.Env, 
    model: ActorCritic, 
    state_dim: int,
    actor_lr: float = 1e-3,
    max_episodes: int = 100, 
    G: int = G, 
    max_steps: int = 512,
    kl_loss_on: bool = True,
    entropy_bonus_on: bool = True
):
    episodes_rewards = []
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

        gspo_update(model, optimizer, group_states, group_actions, group_log_probs, group_advantages, group_masks, kl_loss_on, entropy_bonus_on)

        # print(f"Episode {episode + 1}, Reward: {avg_reward}")
        episodes_rewards.append(group_rewards.sum() / G)
        pbar.set_postfix(episode_reward=episodes_rewards[-1])

    return episodes_rewards

if __name__ == "__main__":
    vec_env = gym.make_vec("CartPole-v1", num_envs=G)
    state_dim, action_dim = vec_env.observation_space.shape[1], vec_env.action_space[0].n
    model = ActorCritic(state_dim, action_dim).to(device)
    # model.load_state_dict(torch.load("./model_parameters/grpo.pt"))
    train(vec_env, model, state_dim, kl_loss_on=True, entropy_bonus_on=True)
    # fig, ax = plt.subplots(2, 2)
    # for idx, kl_loss_on in enumerate([True, False]):
    #     for jdx, entropy_bonus_on in enumerate([True, False]):
    #         episodes_rewards = train(
    #             vec_env=vec_env, 
    #             model=model, 
    #             state_dim=state_dim, 
    #             kl_loss_on=kl_loss_on, 
    #             entropy_bonus_on=entropy_bonus_on
    #         )
    #         ax[idx, jdx].plot(episodes_rewards, label=f"kl_loss_on: {kl_loss_on}, entropy_bonus_on: {entropy_bonus_on}")
    #         ax[idx, jdx].legend()

    # plt.show()
    if not os.path.exists("./figures"):
        os.makedirs("./figures")
    plt.savefig("./figures/gspo.png")

    # if not os.path.exists("./model_parameters"):
    #     os.makedirs("./model_parameters")
    # torch.save(model.state_dict(), "./model_parameters/gspo.pt")