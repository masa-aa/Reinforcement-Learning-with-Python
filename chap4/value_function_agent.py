import random
import argparse
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import gym
from fn_framework import FNAgent, Trainer, Observer
import os


class ValueFunctionAgent(FNAgent):

    def save(self, model_path):
        joblib.dump(self.model, model_path)

    @classmethod
    def load(cls, env, model_path, epsilon=0.0001):
        actions = list(range(env.action_space.n))
        agent = cls(epsilon, actions)
        agent.model = joblib.load(model_path)
        agent.initialized = True
        return agent

    def initialize(self, experiences):
        scaler = StandardScaler()
        # MLPRegressor:価値関数, ノード数が10, 隠れ層を2つ重ねたニューラルネットワーク
        # カートの位置, 加速度, ポールの角度, 角速度を受け取り, 行動を返す関数
        estimator = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1)

        # modelは データの変換 -> 推定 をする奴と定義
        self.model = Pipeline([("scaler", scaler), ("estimator", estimator)])

        # vstackは縦に連結する関数
        states = np.vstack([e.s for e in experiences])
        # 正規化する
        self.model.named_steps["scaler"].fit(states)

        # 1回updateすることでsklearnの例外処理を回避.
        self.update([experiences[0]], gamma=0)
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def estimate(self, s):
        estimated = self.model.predict(s)[0]
        return estimated

    def _predict(self, states):
        if self.initialized:
            predicteds = self.model.predict(states)
        else:
            # 1回目はランダムな値を返す.
            size = len(self.actions) * len(states)
            predicteds = np.random.uniform(size=size)
            predicteds = predicteds.reshape((-1, len(self.actions)))
        return predicteds

    def update(self, experiences, gamma):
        states = np.vstack([e.s for e in experiences])
        n_states = np.vstack([e.n_s for e in experiences])

        # 予測結果
        estimateds = self._predict(states)
        future = self._predict(n_states)

        for i, e in enumerate(experiences):
            reward = e.r
            if not e.d:
                reward += gamma * np.max(future[i])
            # 行動した箇所を更新
            estimateds[i][e.a] = reward

        estimateds = np.array(estimateds)
        states = self.model.named_steps["scaler"].transform(states)
        # partial_fit でTD誤差が小さくなるように調整
        self.model.named_steps["estimator"].partial_fit(states, estimateds)


class CartPoleObserver(Observer):
    """状態はカートの位置, 加速度, ポールの角度, 角速度で, それらを1行4列にreshape"""

    def transform(self, state):
        return np.array(state).reshape((1, -1))


class ValueFunctionTrainer(Trainer):

    def train(self, env, episode_count=200, epsilon=0.1, initial_count=-1,
              render=False):
        actions = list(range(env.action_space.n))
        agent = ValueFunctionAgent(epsilon, actions)
        self.train_loop(env, agent, episode_count, initial_count, render)
        return agent

    def begin_train(self, episode, agent):
        """準備ができたら初期化"""
        agent.initialize(self.experiences)

    def step(self, episode, step_count, agent, experience):
        if self.training:
            batch = random.sample(self.experiences, self.batch_size)
            agent.update(batch, self.gamma)

    def episode_end(self, episode, step_count, agent):
        rewards = [e.r for e in self.get_recent(step_count)]
        self.reward_log.append(sum(rewards))

        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            self.logger.describe("reward", recent_rewards, episode=episode)


def main(play):
    env = CartPoleObserver(gym.make("CartPole-v0"))
    # log_dir = os.path.join(os.path.dirname(__file__), "logs")
    # env = gym.wrappers.Monitor(env, log_dir + "record", video_callable=(lambda ep: ep % 100 == 0))

    trainer = ValueFunctionTrainer()
    path = trainer.logger.path_of("value_function_agent.pkl")

    if play:
        agent = ValueFunctionAgent.load(env, path)
        agent.play(env)
    else:
        trained = trainer.train(env)
        trainer.logger.plot("Rewards", trainer.reward_log,
                            trainer.report_interval)
        trained.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VF Agent")
    # 引数を追加
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")

    # 引数を解析
    args = parser.parse_args()
    main(args.play)
