import numpy as np

class Q_learning(object):

    def __init__(self) -> None:
        self.Q_table = None

        #hyperparameters
        self.alpha: float = 0.1
        self.gamma: float = 0.9
        self.epsilon: float = 0.3
        self.theta: float = 10e-10

    def init_q_table(self, env) -> None:
        self.Q_table = np.zeros((env.get_observation_space().n, env.get_action_space().n))

    def update_q_table(self, state:int, action:int, reward:float, next_state:int) -> None:
        update = self.Q_table[state, action] + self.alpha * (reward + self.gamma * np.max(self.Q_table[next_state]) - self.Q_table[state, action])
        self.Q_table[state, action] = self.Q_table[state, action] + self.alpha * (reward + self.gamma * np.max(self.Q_table[next_state]) - self.Q_table[state, action])

        if np.abs(update - self.Q_table[state, action]) < self.theta:
            return True
        else:
            return False

    def get_action(self, env, state:int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.choice(env.get_action_space().n)
        else:
            return self.Q_table[state].argmax()

    def get_q_table(self) -> np.array:
        return self.Q_table
    
    def save_q_table(self, path:str) -> None:
        np.save(path, self.Q_table)

    def load_q_table(self, path:str) -> None:
        self.Q_table = np.load(path)