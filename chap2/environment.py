from enum import Enum
import numpy as np


class State():
    """状態クラス:セロの位置(行, 列)"""

    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):  # printする時に呼び出される.
        return "<State: [{}, {}]>".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    # https://www.yoheim.net/blog.php?q=20171001 ハッシュ化したり同じか判別する時の判断基準
    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


class Action(Enum):
    """行動クラス:上下左右の移動"""
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


class Environment():

    def __init__(self, grid, move_prob=0.8):
        # grid is 2d-array. Its values are treated as an attribute.

        # 属性の種類は以下の通り
        #  0: 普通のセル
        #  -1: 穴, 落ちると死ぬ (game end)
        #  1: ゴール, 報酬を受け取れるセル (game end)
        #  9: いけないセル (can't locate agent)
        self.grid = grid
        self.agent_state = State()

        # Default reward is minus. Just like a poison swamp.
        # It means the agent has to reach the goal fast!
        self.default_reward = -0.04

        # Agent can move to a selected direction in move_prob.
        # It means the agent will move different direction
        # in (1 - move_prob).
        self.move_prob = move_prob
        self.reset()

    # https://naruport.com/blog/2019/8/27/python-tutorial-class-property-getter-setter/#%E3%82%AF%E3%83%A9%E3%82%B9%E3%81%AE%E3%83%97%E3%83%AD%E3%83%91%E3%83%86%E3%82%A3%EF%BC%88property%EF%BC%89%E3%81%A8%E3%81%AF%EF%BC%9F
    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN,
                Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        """状態列を生成"""
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                # Block cells are not included to the state.
                if self.grid[row][column] != 9:
                    states.append(State(row, column))
        return states

    def transit_func(self, state, action):
        """遷移関数:状態と行動を受け取り, 移動可能なセルとそこへ移動する確率を返す"""
        transition_probs = {}
        if not self.can_action_at(state):
            # 終了しているのでreturn
            return transition_probs

        # 逆方向 風は吹いても逆方向には進まない設定
        opposite_direction = Action(action.value * -1)

        for a in self.actions:
            """遷移確率は,action方向:move_prob, 逆方向:0, 残り:それぞれ(1 - move_prob)/2"""
            prob = 0
            if a == action:
                prob = self.move_prob
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2

            next_state = self._move(state, a)
            # こうしているのは動かない時の対策?
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob

        return transition_probs

    def can_action_at(self, state):
        """actionできるか否か. そのマスが0ならできる"""
        return self.grid[state.row][state.column] == 0

    def _move(self, state, action):
        """実際に動く関数"""
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state = state.clone()

        # actionを実行する (move).
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        """grid外に出た時やブロックセルに行ったときは遷移しない."""
        # Check whether a state is out of the grid.
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state

        # Check whether the agent bumped a block cell.
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    def reward_func(self, state):
        """報酬関数:歩き回ると報酬が減る, 報酬マスに行くとうれしい, ダメージマスに行くと悲しい"""
        reward = self.default_reward
        done = False

        # Check an attribute of next state.
        attribute = self.grid[state.row][state.column]
        if attribute == 1:
            # Get reward! and the game ends.
            reward = 1
            done = True
        elif attribute == -1:
            # Get damage! and the game ends.
            reward = -1
            done = True
        # if done: 終了
        return reward, done

    """ここから外部から操作するための関数"""

    def reset(self):
        # agentを左隅へ移動.
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    def step(self, action):
        """行動を受け取って遷移関数と報酬関数を用いて, 次の遷移先と即時報酬を返す"""
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state

        return next_state, reward, done

    def transit(self, state, action):  # -> next_state, reward, done
        transition_probs = self.transit_func(state, action)
        if len(transition_probs) == 0:
            return None, None, True

        next_states = list(transition_probs.keys())
        probs = list(transition_probs.values())

        # probに基づいてnext_stateを決定する.
        next_state = np.random.choice(next_states, p=probs)
        reward, done = self.reward_func(next_state)
        return next_state, reward, done
