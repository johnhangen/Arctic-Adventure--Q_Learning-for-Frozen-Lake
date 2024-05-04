from src.frozen_lake_env import FrozenLakeEnv
from src.Q_learning import Q_learning


def main() -> None:
    num_episodes = 1_000_000

    env = FrozenLakeEnv()
    env.init_environment()
    env.reset()

    agent = Q_learning()
    agent.init_q_table(env)

    # TODO: it would make sense to add some convergence criteria
    for ep in range(num_episodes):
        action = env.get_action_space().sample()
        _ = env.reset()

        state = 0
        while True:
            action = agent.get_action(env, state)
            S_prime, R, _, _, _ = env.step(action)

            threshhold = agent.update_q_table(env.observation, action, R, S_prime)

            state = S_prime

            if env.terminated or threshhold:
                break
    
    agent.save_q_table('data/Q_table.npy')
    env.plot_rewards()
    env.quit()

if __name__ == '__main__':
    main()
