import random


ROWS, COLS = 4, 4
START = (0, 0)
GOAL = (3, 3)
WALLS = {(1, 1)}
TRAPS = {(2, 1)}

# up, down, left, right
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["up", "down", "left", "right"]


def state_id(position):
    row, col = position
    return row * COLS + col


def step(position, action):
    """Apply an action and return (next_position, reward, done)."""
    row, col = position
    d_row, d_col = ACTIONS[action]
    candidate = (row + d_row, col + d_col)

    if not (0 <= candidate[0] < ROWS and 0 <= candidate[1] < COLS):
        candidate = position
    if candidate in WALLS:
        candidate = position

    if candidate == GOAL:
        return candidate, 20, True
    if candidate in TRAPS:
        return candidate, -10, True
    return candidate, -1, False


def argmax(values):
    """Return a random index among the maximum values."""
    best_value = max(values)
    best_actions = [
        index for index, value in enumerate(values) if value == best_value
    ]
    return random.choice(best_actions)


def train(
    episodes=3000,
    alpha=0.1,
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.05,
):
    q_table = [
        [0.0 for _ in ACTIONS]
        for _ in range(ROWS * COLS)
    ]
    rewards = []

    for _ in range(episodes):
        position = START
        total_reward = 0

        for _ in range(100):
            state = state_id(position)

            if random.random() < epsilon:
                action = random.randrange(len(ACTIONS))
            else:
                action = argmax(q_table[state])

            next_position, reward, done = step(position, action)
            next_state = state_id(next_position)

            next_best = 0 if done else max(q_table[next_state])
            td_target = reward + gamma * next_best
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error

            position = next_position
            total_reward += reward

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)

    return q_table, rewards


def greedy_path(q_table, max_steps=20):
    position = START
    path = [position]

    for _ in range(max_steps):
        state = state_id(position)
        action = max(
            range(len(ACTIONS)),
            key=lambda index: q_table[state][index],
        )
        position, _, done = step(position, action)
        path.append(position)

        if done:
            break

    return path


def print_policy(q_table):
    arrows = ["^", "v", "<", ">"]

    for row in range(ROWS):
        cells = []

        for col in range(COLS):
            position = (row, col)

            if position == GOAL:
                cells.append("G")
            elif position in WALLS:
                cells.append("#")
            elif position in TRAPS:
                cells.append("X")
            else:
                state = state_id(position)
                action = max(
                    range(len(ACTIONS)),
                    key=lambda index: q_table[state][index],
                )
                cells.append(arrows[action])

        print(" ".join(cells))


if __name__ == "__main__":
    random.seed(7)
    q_table, rewards = train()

    average_reward = sum(rewards[-100:]) / 100
    print(
        "Average reward over the final 100 episodes:",
        round(average_reward, 2),
    )
    print("Learned policy:")
    print_policy(q_table)
    print("Greedy path:", greedy_path(q_table))
