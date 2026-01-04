from typing import Hashable, List, Tuple, Iterable
import numpy as np

# types
Probability = float
State = Hashable
Action = Hashable
Reward = float
EndOfEpisode = bool
ValueFunction = np.ndarray
Policy = np.ndarray
Transition = Tuple[Probability, State, Reward, EndOfEpisode]
TransitionProbabilityDict = Dict[State, Dict[Action, List[Transition]]]
StateVisitationFrequencyArray = np.ndarray
StateVisitationProbabilityArray = np.ndarray


class RewardFunction:
    def __init__(self, n_features: int) -> None:
        self.n_features = n_features
        self.theta = np.random.uniform(size=(n_features,))

    def __call__(self, feature_matrix: np.ndarray | int) -> Reward:
        if isinstance(feature_matrix, int):
            feature_matrix = np.eye(self.n_features, 1, feature_matrix).T
        assert feature_matrix.shape == (self.n_features,)
        return feature_matrix.dot(self.theta)

    def update(self, grad: np.ndarray) -> None:
        assert grad.shape == (self.n_features,)
        self.theta += grad


class ValueIteration:
    def __init__(self, n_states: int, n_actions: int, probs: TransitionProbabilityDict):
        self.n_states = n_states
        self.n_actions = n_actions
        self.probs = probs

    def __call__(self, gamma: float, epslion: float, reward_function: RewardFunction | None = None) -> Tuple[ValueFunction, Policy]:
        probs = self.probs
        n_states = self.n_states
        n_actions = self.n_actions
        V = np.zeros(n_states)

        def compute_action_value(state: State) -> np.ndarray:
            A = np.zeros(self.n_actions)
            for action in range(n_actions):
                for prob, next_state, reward, done in probs[state][action]:
                    reward = reward if reward_function is None else reward_function(state)
                    A[action] += prob * (reward + gamma * V[next_state])
            return A

        while True:
            delta = 0
            for state in range(n_states):
                A = compute_action_value(state)
                best_action_value = A.max()
                delta = max(delta, np.abs(best_action_value - V[state]))
                V[state] = best_action_value
            if delta < epslion:
                break

        policy = np.zeros([n_states, n_actions])
        for state in range(n_states):
            A = compute_action_value(state)
            policy[state] = A
        policy -= policy.max(axis=1, keepdims=True)
        max_values = np.broadcast_to(policy.max(axis=1, keepdims=True), policy.shape)
        policy = np.where(policy == max_values, policy, np.NINF)
        policy = np.exp(policy) / np.exp(policy).sum(axis=1, keepdims=True)
        return V, policy


class StateVisitationFrequency:
    def __init__(self, n_states: int, n_actions: int, probs: TransitionProbabilityDict) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.probs = probs

    def __call__(self, policy: Policy, trajectories: np.ndarray) -> StateVisitationFrequencyArray:
        n_states = self.n_states
        n_actions = self.n_actions
        probs = self.probs
        n_trajectories, n_steps = trajectories.shape

        mu = np.zeros((n_steps, n_states))
        for trajectory in trajectories:
            mu[0, trajectory[0]] += 1
        mu /= n_trajectories

        for t in range(1, n_steps):
            for action in range(n_actions):
                for state in range(n_states):
                    for prob, next_state, _, _ in probs[state][action]:
                        mu[t][next_state] += mu[t-1][state] * policy[state][action] * prob

        return mu.sum(axis=0)


def compute_experts_feature(n_features: int, trajectories: np.ndarray) -> StateVisitationProbabilityArray:
    """
    A function that computes the probability of each state feature.
    In the grid world case, the state is represented as a cell.
    In case of learning a formal language, the state could be a word.

    Arguments
    =========
    n_features: int
        Number of features
    trajectories: np.ndarray
        A list of trajectories over n steps. Each element is a feature

    Returns
    =======
    normalized_one_hot_trajectories: np.ndarray
        Probabilities over the features. Higher the probability, more frequently the expert has visited
    """
    if isinstance(trajectories, List):
        trajectories = np.array(trajectories)
    if not isinstance(trajectories, np.ndarray):
        raise TypeError('trajctories must be a np.ndarray')
    if trajectories.ndim != 2:
        raise ValueError('trajctories must be a np.ndarray of dimension 2')

    n_trajectories, n_steps = trajectories.shape
    print('n_trajectories: %i' % (n_trajectories))
    print('n_steps: %i' % (n_steps))

    def one_hot_encoder(array):
        ncols = n_features
        out = np.zeros((array.size, ncols))
        out[np.arange(array.size), array.ravel()] = 1
        out.shape = array.shape + (ncols,)
        return out

    one_hot_trajectories = one_hot_encoder(trajectories)
    assert one_hot_trajectories.shape == (n_trajectories, n_steps, n_features)

    normalized_one_hot_trajectories = one_hot_trajectories.sum(axis=(0, 1)) / n_trajectories
    assert normalized_one_hot_trajectories.shape == (n_features,)

    return normalized_one_hot_trajectories


def train(trajectories: np.ndarray,
          probs: TransitionProbabilityDict
          n_state: int,
          n_action: int,
          n_epochs: int,
          gamma: float, 
          epsilon: float, 
          learning_rate: float) -> RewardFunction:
    """
    Arguments
    =========
    trajectories: np.ndarray
        A list of trajectories over n steps. Each element is a feature
    n_epochs: int
        Number of epochs
    gamma: float
        Discount factor
    epsilon: float
        Epsilon for value iteration
    learning_rate: float
        Learning rate for reward function

    Returns
    =======
    reward_function: RewardFunction
        The learned reward function
    """
    # Requires env to have nS, nA, P
    experts_feature = compute_experts_feature(n_state, trajectories)
    print(experts_feature[:,])

    reward_function = RewardFunction(n_state)
    value_iteration = ValueIteration(n_state, n_action, probs)
    feature_matrix = np.eye(n_state)
    svf = StateVisitationFrequency(n_state, n_action, probs)

    for i in range(n_epochs):
        # 1. Run value iteration with the parameterized reward to get the V and policy pi
        V: ValueFunctionArray, policy: PolicyArray = value_iteration(gamma, epsilon, reward_function)
        # 2. Compute the state visitation frequency P using the policy pi
        P: StateVisitationFrequencyArray = svf(policy, trajectories)
        # 3. Compute the gradient of the reward function
        grad = experts_feature - feature_matrix.T.dot(P)
        # 4. Update the reward function
        reward_function.update(learning_rate * grad)

    return reward_function


if __name__ == '__main__':
    trajectories = [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ]
    train(trajectories, 10, 0.99, 1e-5, 0.1)