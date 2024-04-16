# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        while self.iterations > 0:
            temp_values = self.values.copy()
            states_list = self.mdp.getStates()
            for state in states_list:
                all_actions = self.mdp.getPossibleActions(state)
                vals_list = []
                for action in all_actions:
                    end_states = self.mdp.getTransitionStatesAndProbs(state, action)
                    total = 0
                    for s in end_states:
                        next_state = s[0]
                        probability = s[1]
                        reward = self.mdp.getReward(state, action, next_state)
                        total += (probability * (reward + (self.discount * temp_values[next_state])))
                    vals_list.append(total)
                if len(vals_list) != 0:
                    self.values[state] = max(vals_list)
            self.iterations -= 1



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        end_states_list = self.mdp.getTransitionStatesAndProbs(state, action)
        total_weighted = 0
        for end_state in end_states_list:
            next_state = end_state[0]
            probability = end_state[1]
            reward = self.mdp.getReward(state, action, next_state)
            total_weighted += (probability * (reward + (self.discount * self.values[next_state])))

        return total_weighted


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        all_actions = self.mdp.getPossibleActions(state)
        final_action = ""
        max_value = float("-inf")
        for action in all_actions:
            weighted_value = self.computeQValueFromValues(state, action)
            if (max_value == float("-inf") and action == "") or weighted_value >= max_value:
                final_action = action
                max_value = weighted_value

        return final_action


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def getAllQVals(self, state):
        actions = self.mdp.getPossibleActions(state)  # All possible actions from a state
        qVals = util.Counter()  #  action: qValue pairs

        for action in actions:
            qVals[action] = self.computeQValueFromValues(state, action)
        return qVals

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            index = i % len(self.mdp.getStates())
            state = self.mdp.getStates()[index]
            top_action = self.computeActionFromValues(state)
            if not top_action:
                q_value = 0
            else:
                q_value = self.computeQValueFromValues(state, top_action)
            self.values[state] = q_value



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        priority_queue = util.PriorityQueue()
        all_states_list = self.mdp.getStates()
        predecessors = {}

        for state in all_states_list:
            predecessors[state] = set()

        for state in all_states_list:
            all_actions = self.mdp.getPossibleActions(state)
            for action in all_actions:
                possible_next_states = self.mdp.getTransitionStatesAndProbs(state, action)
                for pos_state in possible_next_states:
                    if pos_state[1] > 0:
                        predecessors[pos_state[0]].add(state)

        for state in all_states_list: 
            all_q_values = self.getAllQVals(state)
            if len(all_q_values) > 0:
                max_q = all_q_values[all_q_values.argMax()]
                diff = abs(self.values[state] - max_q)
                priority_queue.push(state, -diff)

        for i in range(self.iterations):
            if priority_queue.isEmpty():
                return None
            state = priority_queue.pop()
            all_q_values = self.getAllQVals(state)
            max_q = all_q_values[all_q_values.argMax()]
            self.values[state] = max_q
            for pred_state in predecessors[state]:
                pred_q_values = self.getAllQVals(pred_state)
                max_q = pred_q_values[pred_q_values.argMax()]
                diff = abs(self.values[pred_state] - max_q)
                if diff > self.theta:
                    priority_queue.update(pred_state, -diff)

