import gym
import monte_carlo
import q_learning

if __name__ == '__main__':
    env = gym.make("FrozenLakeEasy-v0")
    monte_carlo.train(env=env)
    q_learning.train(env=env)
