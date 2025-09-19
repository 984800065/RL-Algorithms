import gymnasium as gym
import torch
from torch import nn
from tqdm import tqdm


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


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
    # print(group_rewards[-1])
    # print(group_masks[-1])
    r_mean = group_rewards.mean()
    r_std = torch.sqrt((group_rewards - r_mean).pow(2).mean())
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
    minibatch_size: int = 64,
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
                # ref_ratio = torch.exp(ref_group_log_probs[g, batch_idx] - log_probs)
                # KL_loss = beta * (ref_ratio - torch.log(ref_ratio) - 1).mean()
                entropy_bonus = -entropy_coef * entropy
                loss = clip_loss + entropy_bonus
                loss /= G

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
def train(
    env: gym.Env, 
    model: ActorCritic, 
    actor_lr: float = 1e-4, 
    max_episodes: int = 1000, 
    G: int = 32, 
    max_steps: int = 512
):
    optimizer = torch.optim.AdamW([
        {"params": model.actor.parameters(), "lr": actor_lr},
    ])
    state_dim = env.observation_space.shape[0]

    pbar = tqdm(range(max_episodes))
    for episode in pbar:
        # 采样G组数据
        model.eval()
        group_states = torch.zeros((G, max_steps, state_dim), dtype=torch.float, device=device)
        group_actions = torch.zeros((G, max_steps), dtype=torch.long, device=device)
        group_log_probs = torch.zeros((G, max_steps), dtype=torch.float, device=device)
        group_rewards = torch.zeros((G, max_steps), dtype=torch.float, device=device)
        group_masks = torch.ones((G, max_steps), dtype=torch.float, device=device)
        with torch.no_grad():
            for g in range(G):
                state, _ = env.reset()
                for step in range(max_steps):
                    state_tensor = torch.tensor(state, dtype=torch.float).to(device)
                    probs: torch.Tensor = model(state_tensor)
                    dis = torch.distributions.Categorical(probs)
                    action = dis.sample()  # shape: ()
                    next_state, reward, done, _, _ = env.step(action.item())

                    group_states[g, step] = state_tensor
                    group_actions[g, step] = action
                    group_log_probs[g, step] = dis.log_prob(action)
                    group_rewards[g, step] = reward

                    state = next_state
                    if done:
                        group_masks[g, step + 1:] = 0
                        break

        group_advantages = get_group_process_advantages(group_rewards, group_masks)

        grpo_update(model, optimizer, group_states, group_actions, group_log_probs, group_advantages, group_masks)

        # print(f"Episode {episode + 1}, Reward: {avg_reward}")
        pbar.set_postfix(group_reward=group_rewards.sum() / G)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.n
    model = ActorCritic(state_dim, action_dim).to(device)
    tmp_state_dict = torch.load("./model_parameters/ppo.pt")
    # model.actor[0].load_state_dict({
    #     "weight": tmp_state_dict["shared.0.weight"],
    #     "bias": tmp_state_dict["shared.0.bias"]
    # })
    # model.actor[2].load_state_dict({
    #     "weight": tmp_state_dict["actor.0.weight"],
    #     "bias": tmp_state_dict["actor.0.bias"]
    # })
    # model.actor[4].load_state_dict({
    #     "weight": tmp_state_dict["actor.2.weight"],
    #     "bias": tmp_state_dict["actor.2.bias"]
    # })
    train(env, model)