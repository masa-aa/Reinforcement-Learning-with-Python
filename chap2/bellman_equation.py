# Valueベースのベルマン方程式の解 全探索する.
def V(s, gamma=0.99):
    """価値 V の定義 P.34の最後の式　O(2**n)"""
    V = R(s) + gamma * max_V_on_next_state(s)
    return V


def R(s):
    """即時報酬 R(s)"""
    if s == "happy_end":
        return 1
    elif s == "bad_end":
        return -1
    else:
        return 0


def max_V_on_next_state(s):
    """max(V(s'))"""
    # ゲームが終了した場合, 期待値は0.
    if s in ["happy_end", "bad_end"]:
        return 0

    actions = ["up", "down"]
    values = []
    for a in actions:
        transition_probs = transit_func(s, a)
        v = 0
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            v += prob * V(next_state)  # 再帰
        values.append(v)
    return max(values)


def transit_func(s, a):
    """
    遷移関数
    ex: (s = 'state', a = 'up') => 'state_up'
        (s = 'state_up', a = 'down') => 'state_up_down'
    """
    # print(s)
    actions = s.split("_")[1:]
    LIMIT_GAME_COUNT = 5
    HAPPY_END_BORDER = 4
    MOVE_PROB = 0.9

    def next_state(state, action):
        """state_up_up_down_downみたいな感じで返す."""
        return "_".join([state, action])

    if len(actions) == LIMIT_GAME_COUNT:
        up_count = sum([a == "up" for a in actions])
        state = "happy_end" if up_count >= HAPPY_END_BORDER else "bad_end"
        prob = 1.0
        return {state: prob}
    else:
        opposite = "up" if a == "down" else "down"
        return {
            next_state(s, a): MOVE_PROB,
            next_state(s, opposite): 1 - MOVE_PROB
        }


if __name__ == "__main__":
    # 期待値をとるので値は決定的
    print(V("state"))
    print(V("state_up_up"))
    print(V("state_down_down"))
