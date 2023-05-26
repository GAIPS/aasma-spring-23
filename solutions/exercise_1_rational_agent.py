import argparse
import numpy as np
from collections import defaultdict

from aasma import Agent
from aasma.wrappers import SingleAgentWrapper
from aasma.utils import compare_results_learning
from aasma.simplified_predator_prey import SimplifiedPredatorPrey


def train_eval_loop_single(train_environment, eval_environment, agent, n_evaluations, n_training_episodes, n_eval_episodes):

    print(f"Train-Eval Loop for {agent.name}\n")

    results = np.zeros((n_evaluations, n_eval_episodes))

    for evaluation in range(n_evaluations):

        print(f"\tIteration {evaluation+1}/{n_evaluations}")

        # Train
        print(f"\t\tTraining {agent.name} for {n_training_episodes} episodes.")
        agent.train()   # Enables training mode
        run_single(train_environment, agent, n_training_episodes)

        # Eval
        print(f"\t\tEvaluating {agent.name} for {n_training_episodes} episodes.")
        agent.eval()    # Disables training mode
        results[evaluation] = run_single(eval_environment, agent, n_eval_episodes)
        print(f"\t\tAverage Steps To Capture: {round(results[evaluation].mean(), 2)}")
        print()

    return results


def run_single(environment, agent, n_episodes):

    results = np.zeros(n_episodes)

    for episode in range(n_episodes):

        steps = 0
        terminal = False
        observation = environment.reset()
        while not terminal:
            steps += 1
            agent.see(observation)
            action = agent.action()
            next_observation, reward, terminal, info = environment.step(action)
            if agent.training:
                agent.next(observation, action, next_observation, reward, terminal, info)
            observation = next_observation

        environment.close()
        results[episode] = steps

    return results


class QLearning(Agent):

    def __init__(self, n_actions, learning_rate=0.3, discount_factor=0.99, exploration_rate=0.15, initial_q_values=0.0):
        self._Q = defaultdict(lambda: np.ones(n_actions) * initial_q_values)
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._exploration_rate = exploration_rate
        self._n_actions = n_actions
        super(QLearning, self).__init__("Q-Learning")

    def action(self, explore=True):

        q_values = self._Q[tuple(self.observation)]

        if not self.training or (self.training and np.random.uniform(0, 1) > self._exploration_rate):
            # Exploit
            actions = np.argwhere(q_values == np.max(q_values)).reshape(-1)
        else:
            # Explore
            actions = range(self._n_actions)

        return np.random.choice(actions)

    def next(self, observation, action, next_observation, reward, terminal, info):

        x, a, r, y = tuple(self.observation), action, reward, tuple(next_observation)
        alpha, gamma = self._learning_rate, self._discount_factor

        Q_xa, Q_y = self._Q[x][a], self._Q[y]
        max_Q_ya = max(Q_y)

        self._Q[x][a] = Q_xa + alpha * (r + gamma * max_Q_ya - Q_xa)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-training", type=int, default=100)
    parser.add_argument("--episodes-per-evaluation", type=int, default=32)
    parser.add_argument("--evaluations", type=int, default=10)
    opt = parser.parse_args()

    # 1 - Setup environments
    # We set up two instances of the same environment
    # One for training, one for evaluation
    train_environment = SimplifiedPredatorPrey(
        grid_shape=(7, 7),
        n_agents=1, n_preys=1,
        max_steps=100, required_captors=1
    )
    eval_environment = SimplifiedPredatorPrey(
        grid_shape=(7, 7),
        n_agents=1, n_preys=1,
        max_steps=100, required_captors=1
    )
    train_environment = SingleAgentWrapper(train_environment, agent_id=0)
    eval_environment = SingleAgentWrapper(eval_environment, agent_id=0)

    # 2 - Setup agent
    agent = QLearning(train_environment.action_space.n)

    # 3 - Evaluate agent
    results = {
        agent.name: train_eval_loop_single(
            train_environment, eval_environment, agent,
            opt.evaluations, opt.episodes_per_training, opt.episodes_per_evaluation)
    }

    # 4 - Compare results
    compare_results_learning(results, title="Q-Learning Agent on 'Predator Prey' Environment", colors=["blue"])

