import argparse

from aasma.wrappers import SingleAgentWrapper
from aasma.utils import compare_results_learning
from aasma.simplified_predator_prey import SimplifiedPredatorPrey

from lab1_solutions.exercise_1_single_random_agent import RandomAgent
from lab1_solutions.exercise_2_single_random_vs_greedy import GreedyAgent

from solutions.exercise_1_rational_agent import QLearning, train_eval_loop_single


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

    # 2 - Setup agents
    agents = [
        QLearning(train_environment.action_space.n),
        RandomAgent(train_environment.action_space.n),
        GreedyAgent(agent_id=0, n_agents=1)
    ]

    # 3 - Evaluate agents
    results = {}
    for agent in agents:
        result = train_eval_loop_single(train_environment, eval_environment, agent,
                                 opt.evaluations, opt.episodes_per_training, opt.episodes_per_evaluation)
        results[agent.name] = result

    # 4 - Compare results
    compare_results_learning(results,
                             title="Agents on 'Predator Prey' Environment",
                             colors=["blue", "orange", "green"])