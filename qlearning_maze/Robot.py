import random
from Maze import Maze
import numpy as np
from Runner import Runner

class Robot(object):

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.5):

        self.maze = maze
        self.valid_actions = self.maze.valid_actions
        self.state = None
        self.action = None

        # Set Parameters of the Learning Robot
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.t = 0
        self.nA = 4

        self.Qtable = {}
        self.reset()

    def reset(self):
        """
        Reset the robot
        """
        self.state = self.sense_state()
        self.create_Qtable_line(self.state)

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing

    def update_parameter(self):
        """
        Some of the paramters of the q learning robot can be altered,
        update these parameters when necessary.
        """
        if self.testing:
            # TODO 1. No random choice when testing
            pass
        else:
            # TODO 2. Update parameters when learning
            self.epsilon = 1.0/(self.t/100+1.0)

        return self.epsilon

    def sense_state(self):
        """
        Get the current state of the robot. In this
        """

        # TODO 3. Return robot's current state
        return self.maze.sense_robot()

    def create_Qtable_line(self, state):
        """
        Create the qtable with the current state
        """
        # TODO 4. Create qtable with current state
        # Our qtable should be a two level dict,
        # Qtable[state] ={'u':xx, 'd':xx, ...}
        # If Qtable[state] already exits, then do
        # not change it.
        if state not in self.Qtable.keys():
            self.Qtable[state] = {'u':0, 'r':0, 'd':0, 'l':0}

    def choose_action(self):
        """
        Return an action according to given rules
        """
        def is_random_exploration():

            # TODO 5. Return whether do random choice
            # hint: generate a random number, and compare
            # it with epsilon
            return self.epsilon > random.random()

        if self.learning:
            if is_random_exploration():
                # TODO 6. Return random choose aciton
                act = np.random.choice(np.arange(len(self.valid_actions)), p=self.epsilon_greedy_probs(self.Qtable[self.state], self.epsilon, self.nA))
                return self.valid_actions[act]
            else:
                # TODO 7. Return action with highest q value
                Qs = self.Qtable[self.state]
                return max(Qs,key=Qs.get)
        elif self.testing:
            # TODO 7. choose action with highest q value
            Qs = self.Qtable[self.state]
            return max(Qs, key=Qs.get)
        else:
            # TODO 6. Return random choose aciton
            act = np.random.choice(np.arange(self.nA),
                                   p=get_probs(self.Qtable[self.state], self.epsilon, self.nA))
            return self.valid_actions[act]

    def update_Qtable(self, r, action, next_state):
        """
        Update the qtable according to the given rule.
        Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))
        """
        if self.learning:
            # TODO 8. When learning, update the q table according
            # to the given rules
            Qnt = self.Qtable[next_state]
            Qsa = self.Qtable[self.state][action]
            maxQnt = self.Qtable[next_state][max(Qnt,key=Qnt.get)]
            self.Qtable[self.state][action] = Qsa + self.alpha*(r + (self.gamma * (maxQnt) - Qsa))
            #print(self.state,':',self.Qtable[self.state])

    def update(self):
        """
        Describle the procedure what to do when update the robot.
        Called every time in every epoch in training or testing.
        Return current action and reward.
        """
        self.state = self.sense_state() # Get the current state
        self.create_Qtable_line(self.state) # For the state, create q table line

        action = self.choose_action() # choose action for this state
        reward = self.maze.move_robot(action) # move robot for given action

        next_state = self.sense_state() # get next state
        self.create_Qtable_line(next_state) # create q table line for next state

        if self.learning and not self.testing:
            self.t = self.t + 1
            self.update_Qtable(reward, action, next_state) # update q table
            self.update_parameter() # update parameters

        return action, reward

    def epsilon_greedy_probs(self, Q_s, epsilon, nA):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(nA) * epsilon / nA
        best_a = np.argmax(Q_s)
        policy_s[best_a] = 1 - epsilon + (epsilon / nA)
        #print("policy_s:",policy_s)
        return policy_s


'''mm = Maze(maze_size=(10,12),trap_number=6)
robot = Robot(mm)
robot.set_status(learning=True,testing=False)
print(robot.update())

epoch = 20

epsilon0 = 0.5
alpha = 0.5
gamma = 0.9

maze_size = (6,6)
trap_number = 1

g = Maze(maze_size=maze_size,trap_number=trap_number)
r = Robot(g,alpha=alpha, epsilon0=epsilon0, gamma=gamma)
r.set_status(learning=True)
runner = Runner(r, g)
runner.run_training(epoch, display_direction=True)'''
