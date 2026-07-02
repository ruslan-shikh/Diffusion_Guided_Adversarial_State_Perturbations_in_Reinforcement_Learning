"""Step 1 sanity: load the shipped Freeway victim and play one clean (no-attack) episode.
Expected: total reward near the paper's 'DQN-No Attack' Freeway = ~34. Run from repo root with the 3.9 venv.
"""
import sys
sys.path.append("src")
import torch
from envs import make_atari_env_test
from model import model_setup


def main():
    device = torch.device("cuda:0")
    env_id = "FreewayNoFrameskip-v4"
    env = make_atari_env_test(env_id, num_envs=1, device=device,
                              done_on_life_loss=False, size=84, max_episode_steps=None)
    print("obs space:", env.observation_space.shape, "| num_actions:", env.num_actions)

    policy = model_setup(env_id, env, robust_model=False, logger=None,
                         use_cuda=True, dueling=True, model_width=1).to(device)
    sd = torch.load("src/pre_trained/Freeway-natural.model", map_location=device)
    policy.features.load_state_dict(sd)
    policy.eval()
    print("victim loaded OK")

    for ep in range(3):
        obs, _ = env.reset(seed=ep)
        total, steps, done = 0.0, 0, False
        while not done and steps < 20000:
            with torch.no_grad():
                action = policy.act(obs, epsilon=0)
            obs, rew, end, trunc, info = env.step(action)
            total += float(rew.sum().item())
            done = bool(end.any() or trunc.any())
            steps += 1
        print(f"episode {ep}: reward={total:.1f}  steps={steps}")


if __name__ == "__main__":
    main()
