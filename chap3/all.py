import gym
import monte_carlo
import q_learning
import sarsa
import actor_critic


if __name__ == '__main__':
    env = gym.make("FrozenLakeEasy-v0")
    env.render()
    monte_carlo.train(env=env)
    q_learning.train(env=env)
    sarsa.train(env=env)
    actor_critic.train(env=env)
