import argparse
import itertools
import numpy as np
from gym import Wrapper
from typing import Sequence

from aasma.simplified_predator_prey import SimplifiedPredatorPrey
from aasma.utils import compare_results_learning

from lab1_solutions.exercise_2_single_random_vs_greedy import GreedyAgent

from solutions.exercise_1_rational_agent import QLearning, train_eval_loop_single


def train_eval_loop_multi(train_environment, eval_environment, team, agents, n_evaluations, n_training_episodes, n_eval_episodes):

    print(f"Train-Eval Loop for {team}\n")

    results = np.zeros((n_evaluations, n_eval_episodes))

    for evaluation in range(n_evaluations):

        print(f"\tIteration {evaluation+1}/{n_evaluations}")

        # Train
        print(f"\t\tTraining {team} for {n_training_episodes} episodes.")
        for agent in agents: agent.train()
        run_multi(train_environment, agents, n_training_episodes)

        # Eval
        print(f"\t\tEvaluating {team} for {n_training_episodes} episodes.")
        for agent in agents: agent.eval()
        results[evaluation] = run_multi(eval_environment, agents, n_eval_episodes)
        print(f"\t\tAverage Steps To Capture: {round(results[evaluation].mean(), 2)}")
        print()

    return results


def run_multi(environment, agents, n_episodes):

    results = np.zeros(n_episodes)

    for episode in range(n_episodes):

        steps = 0
        terminals = [False for _ in range(len(agents))]
        observations = environment.reset()

        while not all(terminals):
            steps += 1
            for observations, agent in zip(observations, agents):
                agent.see(observations)
            actions = [agent.action() for agent in agents]
            next_observations, rewards, terminals, info = environment.step(actions)
            for a, agent in enumerate(agents):
                if agent.training:
                    agent.next(observations[a], actions[a], next_observations[a], rewards[a], terminals[a], info)
            observations = next_observations
        results[episode] = steps

        environment.close()

    return results


class JointActionWrapper(Wrapper):

    """ A Wrapper for centralized multi-agent environments.

    * Allows a single agent to control all agents via a global joint-action.
    * Reduces the N action spaces (where N is the number of agents) to a single joint-action space.

    Example
    -------
    >> N = 2
    >> Actions Agent 0 = Actions Agent 1 = [ Move (0), Stay (1) ]

    | Action 1 | Action 2 | Team Action |
    |----------|----------|--------------|
    | 0        | 0        | 0            |
    | 0        | 1        | 1            |
    | 1        | 0        | 2            |
    | 1        | 1        | 3            |
    """

    def __init__(self, env):

        super(JointActionWrapper, self).__init__(env)

        self.n_agents = env.n_agents

        self.action_spaces = [list(range(env.action_space[a].n)) for a in range(self.n_agents)]
        self.joint_action_space = list(itertools.product(*self.action_spaces))

        self.action_meanings = [env.get_action_meanings(a) for a in range(self.n_agents)]
        self.joint_action_meanings = list(itertools.product(*self.action_meanings))

        self.n_joint_actions = len(self.joint_action_meanings)

    def reset(self):
        observations = super(JointActionWrapper, self).reset()
        observation = observations[0]   # For the predator-prey domain, the observations are shared.
        return observation

    def step(self, joint_action: int):

        individual_actions: Sequence[int] = self.joint_action_space[joint_action]
        next_observations, rewards, terminals, info = super(JointActionWrapper, self).step(individual_actions)

        next_observation = next_observations[0]    # For the predator-prey domain, the observations are shared.
        equal_rewards = all(rewards[0] == reward for reward in rewards)
        assert equal_rewards, "Multi-Agent RL requires same reward signal for all agents"
        reward = rewards[0]
        terminal = all(terminals)

        return next_observation, reward, terminal, info

    def get_action_meanings(self):
        return self.team_action_meanings


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-training", type=int, default=100)
    parser.add_argument("--episodes-per-evaluation", type=int, default=64)
    parser.add_argument("--evaluations", type=int, default=10)
    opt = parser.parse_args()

    # 1 - Setup environments
    # We set up two instances of the same environment
    # One for training, one for evaluation
    train_environment = SimplifiedPredatorPrey(grid_shape=(5, 5), n_agents=2, n_preys=1, max_steps=100, required_captors=2)
    eval_environment = SimplifiedPredatorPrey(grid_shape=(5, 5), n_agents=2, n_preys=1, max_steps=100, required_captors=2)

    # 2 - Setup centralized multi-agent learner
    # This can be represented by a single agent controlling a joint action space
    joint_train_environment = JointActionWrapper(train_environment)
    joint_eval_environment = JointActionWrapper(eval_environment)
    centralized_multi_agent_learner = QLearning(joint_train_environment.n_joint_actions)

    # 3 - Setup decentralized learners
    decentralized_single_agent_learners = [
        QLearning(train_environment.action_space[0].n),
        QLearning(train_environment.action_space[1].n)
    ]

    # 4 - Reactive Teams
    greedy_reactive_team = [
        GreedyAgent(0, 2),
        GreedyAgent(1, 2)
    ]

    # 4 - Evaluate
    results = {

        # Note here we can treat Central-Agent RL as a single agent system,
        # Since the environment now accepts team_actions from a single algorithm
        "1 Centralized Multi-Agent Learner": train_eval_loop_single(
            joint_train_environment, joint_eval_environment, centralized_multi_agent_learner,
            opt.evaluations, opt.episodes_per_training, opt.episodes_per_evaluation),

        "2 Decentralized Single-Agent Learners": train_eval_loop_multi(
            train_environment, eval_environment, "QLearning Team", decentralized_single_agent_learners,
            opt.evaluations, opt.episodes_per_training, opt.episodes_per_evaluation),

        "2 Reactive Agents (Greedy)": train_eval_loop_multi(
            train_environment, eval_environment, "Greedy Team", greedy_reactive_team,
            opt.evaluations, opt.episodes_per_training, opt.episodes_per_evaluation),

    }

    # 5 - Compare results
    compare_results_learning(results,
                             title="Multi vs Single-Agent Learning 'Predator Prey' Environment\nReactive Agents for comparison",
                             colors=["blue", "lightblue", "green"])
