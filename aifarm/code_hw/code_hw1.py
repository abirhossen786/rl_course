from typing import List, Dict, Optional, Tuple
from environments.environment_abstract import Environment, State
from environments.farm_grid_world import FarmGridWorld
import numpy as np
from visualizer.farm_visualizer import InteractiveFarm
import time


def update_dp(viz: InteractiveFarm, state_values, policy):
    viz.set_state_values(state_values)
    viz.set_policy(policy)
    viz.window.update()


def update_model_free(viz: InteractiveFarm, state, action_values):
    viz.set_action_values(action_values)
    viz.board.delete(viz.agent_img)
    viz.agent_img = viz.place_imgs(viz.board, viz.robot_pic, [state.agent_idx])[0]
    viz.window.update()


def policy_evaluation(env: Environment, states: List[State], state_values: Dict[State, float],
                      policy: Dict[State, List[float]], discount: float, cutoff: float) -> Dict[State, float]:
    """
    @param env: environment
    @param states: all states in the state space
    @param state_values: dictionary that maps states to values
    @param policy: dictionary that maps states to a list of probabilities of taking each action
    @param discount: the discount factor
    @param cutoff: while delta > cutoff, continue policy evaluation

    @return: state values for each state when following the given policy. This must be a dictionary that maps states
    to values
    """

    itr: int = 0
    delta: float = np.inf
    while delta > cutoff:
        delta = 0.0
        for state in states:
            val_curr: float = state_values[state]

            state_value_new: float = 0.0
            for action in env.get_actions():
                expected_reward, next_states, probs = env.state_action_dynamics(state, action)

                expected_value_next: float = 0.0
                for next_state, prob in zip(next_states, probs):
                    expected_value_next += prob * state_values[next_state]

                action_value = expected_reward + discount * expected_value_next
                state_value_new += policy[state][action] * action_value

            state_values[state] = state_value_new

            delta_state: float = abs(val_curr - state_values[state])
            delta = max(delta, delta_state)

        print("Policy evaluation: %i, delta: %E" % (itr, delta))
        itr += 1

    return state_values


def policy_improvement(env: Environment, states: List[State], state_values: Dict[State, float],
                       discount: float) -> Dict[State, List[float]]:
    """ Return policy that behaves greedily with respect to value function

    @param env: environment
    @param states: all states in the state space
    @param state_values: dictionary that maps states to values
    @param discount: the discount factor

    @return: the policy that behaves greedily with respect to the value function. This must be a dictionary that maps
    states to list of floats. The list of floats is the probability of taking each action.
    """
    policy_new: Dict[State, List[float]] = dict()
    for state in states:
        action_values: List[float] = []
        for action in env.get_actions():
            expected_reward, next_states, probs = env.state_action_dynamics(state, action)

            expected_value_next: float = 0.0
            for next_state, prob in zip(next_states, probs):
                expected_value_next += prob * state_values[next_state]

            action_val = expected_reward + discount * expected_value_next

            action_values.append(action_val)

        action = int(np.argmax(action_values))
        policy_new[state] = [0, 0, 0, 0]
        policy_new[state][action] = 1

    return policy_new


def policy_iteration(env: FarmGridWorld, states: List[State], state_values: Dict[State, float],
                     policy: Dict[State, List[float]], discount: float, policy_eval_cutoff: float,
                     viz: Optional[InteractiveFarm]) -> Tuple[Dict[State, float], Dict[State, List[float]]]:
    """
    @param env: environment
    @param states: all states in the state space
    @param state_values: dictionary that maps states to values
    @param policy: dictionary that maps states to a list of probabilities of taking each action
    @param discount: the discount factor
    @param policy_eval_cutoff: the cutoff for policy evaluation
    @param viz: optional visualizer

    @return: the state value function and policy found by policy iteration
    """
    if viz is not None:
        update_dp(viz, state_values, policy)

    wait = 0.1
    policy_changed: bool = True
    itr: int = 0
    while policy_changed:
        # policy evaluation
        state_values = policy_evaluation(env, states, state_values, policy, discount, policy_eval_cutoff)
        policy_new = policy_improvement(env, states, state_values, discount)

        # check for convergence
        policy_changed = policy != policy_new
        policy = policy_new
        itr += 1

        # visualize
        print("Policy iteration itr: %i" % itr)
        if (wait > 0.0) and (viz is not None):
            update_dp(viz, state_values, policy)
            time.sleep(wait)

    update_dp(viz, state_values, policy)

    return state_values, policy


def value_iteration(env: Environment, states: List[State], state_values: Dict[State, float],
                    discount: float, cutoff: float,
                    viz: Optional[InteractiveFarm]) -> Tuple[Dict[State, float], Dict[State, List[float]]]:
    """
    @param env: environment
    @param states: all states in the state space
    @param state_values: dictionary that maps states to values
    @param discount: the discount factor
    @param cutoff: while delta > cutoff, continue value iteration
    @param viz: optional visualizer

    @return: the state value function and policy found by value iteration
    """
    delta: float = np.inf
    wait = 0.1
    itr: int = 0

    while delta > cutoff:
        change_itr: float = 0.0
        for state_idx, state in enumerate(states):
            val_curr: float = state_values[state]

            action_vals: List[float] = []
            for action in env.get_actions():
                expected_reward, next_states, probs = env.state_action_dynamics(state, action)

                expected_val: float = 0.0
                for next_state, prob in zip(next_states, probs):
                    expected_val += prob * state_values[next_state]

                action_val = expected_reward + discount * expected_val

                action_vals.append(action_val)

            state_values[state]: float = max(action_vals)

            change_state: float = abs(val_curr - state_values[state])
            change_itr = max(change_itr, change_state)

        print("Value iteration itr: %i, change_itr: %.2E, min: %f" % (itr, change_itr, np.min(list(state_values.values()))))
        if (wait > 0.0) and (viz is not None):
            policy = policy_improvement(env, states, state_values, discount)
            update_dp(viz, state_values, policy)
            time.sleep(wait)

        itr += 1

        delta = change_itr

    policy = policy_improvement(env, states, state_values, discount)

    return state_values, policy


def q_learning(env: Environment, action_values: Dict[State, List[float]], epsilon: float, learning_rate: float,
               discount: float, num_episodes: int, viz: Optional[InteractiveFarm]) -> Dict[State, List[float]]:
    """
    @param env: environment
    @param action_values: dictionary that maps states to their action values (list of floats)
    @param epsilon: epsilon-greedy policy
    @param learning_rate: learning rate
    @param discount: the discount factor
    @param num_episodes: number of episodes for learning
    @param viz: optional visualizer

    @return: the action value function found by Q-learning
    """

    state = env.sample_start_state()
    update_model_free(viz, state, action_values)

    episode_num: int = 0
    episode_step: int = 0
    wait = 0.0
    print("Q-learning, episode %i" % episode_num)
    while episode_num < num_episodes:
        if env.is_terminal(state) or (episode_step > 50):
            episode_num = episode_num + 1
            episode_step = 0
            state = env.sample_start_state()

            print("Q-learning, episode %i" % episode_num)

        if np.random.rand(1)[0] < epsilon:
            num_actions: int = len(action_values[state])
            action: int = np.random.choice(num_actions)
        else:
            action: int = int(np.argmax(action_values[state]))

        state_next, reward = env.sample_transition(state, action)

        action_val_curr_state: float = action_values[state][action]
        action_val_next_state: float = max(action_values[state_next])

        td: float = reward + discount * action_val_next_state - action_val_curr_state
        action_values[state][action] += learning_rate * td

        state = state_next

        if wait > 0.0:
            update_model_free(viz, state, action_values)
            time.sleep(wait)

        episode_step += 1

    return action_values
