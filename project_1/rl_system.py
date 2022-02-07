import copy
import random

# import doodler as DDL
import matplotlib.pyplot as plt
import numpy as np
import tflow.kdtflowclasses as KDTFC
from tensorflow import keras as KER

from parameters import critic_type, episodes, lr_critic, lr_actor, \
    discount_factor_critic, discount_factor_actor, eligibility_decay_critic, eligibility_decay_actor
from simworld import SimWorld


class Actor:
    def __init__(self, critic, sim_world):
        self.critic = critic
        self.sim_world = sim_world
        self.policy = dict()
        self.eligibility = dict()

    def get_best_action(self, state):
        actions = self.policy[state]
        highest_value = max(actions.values())
        best_actions = []
        for key, value in actions.items():
            if value == highest_value:
                best_actions.append(key)
        return random.choice(best_actions)

    def initialize_policy_function(self, all_states):
        for s in all_states:
            self.policy[s] = dict()
            all_actions_for_state = self.sim_world.get_possible_actions_from_state(
                s)
            for a in all_actions_for_state:
                self.policy[s][a] = 0

    def initialize_eligibility_function(self):
        self.eligibility = copy.deepcopy(self.policy)
        # set all action values in each state to 0
        for state in self.eligibility:
            self.eligibility[state] = dict.fromkeys(self.eligibility[state], 0)

    def set_eligibility(self, state, action, value):
        if value is None:
            self.eligibility[state][action] *= discount_factor_actor * eligibility_decay_actor
        else:
            self.eligibility[state][action] = value

    def set_policy(self, state, action):
        self.policy[state][action] += lr_actor * self.critic.get_TD_error() * self.eligibility[state][action]


class CriticTable:
    def __init__(self, type, sim_world):
        self.type = type  # How should we handle the type case
        self.sim_world = sim_world
        self.V = dict()
        self.eligibility = dict()
        self.TD_error = 0

    def get_state(self):
        return self.sim_world.get_state()

    def get_TD_error(self):
        return self.TD_error

    def initialize_value_function(self, all_states):
        for s in all_states:
            self.V[s] = random.randint(0, 10)

    # should this be done differently?
    def initialize_eligibility_function(self, all_states):
        for s in all_states:
            self.eligibility[s] = 0

    def set_TD_error(self, r, state, new_state):
        self.TD_error = r + discount_factor_critic * self.V[new_state] - self.V[state]

    def set_eligibility(self, state, value):
        if value is None:
            self.eligibility[state] *= discount_factor_critic * eligibility_decay_critic
        else:
            self.eligibility[state] = value

    def set_value_for_state(self, state):
        self.V[state] += lr_critic * self.TD_error * self.eligibility[state]


class CriticANN:
    def __init__(self):
        pass

    # Generate a convolution network.
    def gencon(num_classes=10, lrate=0.01, opt='SGD', loss='categorical_crossentropy', act='relu'):
        opt = eval('KER.optimizers.' + opt)
        loss = eval('KER.losses.' + loss)
        model = KER.models.Sequential()  # The model can now be built sequentially from input to output
        # The first layer can include the dims of the upstream layer (input_shape = in_dims)
        #   Otherwise, this part of the model will be configured during the call to model.fit().
        model.add(KER.layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation=act))
        model.add(KER.layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
        model.add(KER.layers.Conv2D(32, kernel_size=(3, 3), activation=act, strides=(1, 1)))
        model.add(KER.layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
        model.add(KER.layers.Flatten())
        model.add(KER.layers.Dense(100, activation=act))
        model.add(KER.layers.Dense(num_classes, activation='softmax'))
        model.compile(optimizer=opt(lr=lrate), loss=loss, metrics=[KER.metrics.categorical_accuracy])
        return model

    # LEarn to COunt using the 2-d doodle diagrams.  Range of image counts = (im0, im1)

    def leco2(epochs=100, im0=0, im1=8, ncases=500, dims=(50, 50), gap=1, vf=0.2, lrate=0.01, mbs=16, act='relu'):
        tlen = im1 - im0 + 1  # length of target vectors
        in_shape = list(dims) + [1]  # 1 = no. input channels

        # A simple, local function for generating 1-hot target vectors from an integer. E.g. 2 => (0 0 1 0 0 ...)
        def gentarget(k):
            targ = [0] * tlen
            targ[k] = 1
            return targ

        # Create a Doodler object for creating the doodle images
        # d = DDL.Doodler(rows=dims[0], cols=dims[1], gap=gap, multi=True)
        # cases = d.gen_random_cases(ncases, image_types=['ball', 'box', 'frame', 'ring'], flat=False,

        wr = [0.1, 0.25], hr = [0.1, 0.25], figcount = (im0, im1))

        # Pull the inputs and targets out of the cases
        inputs = np.array([np.array(c[0]).reshape(in_shape).astype(np.float32) for c in cases])
        targets = np.array([gentarget(c[1]) for c in cases])

        # Generate the convolutional neural network
        nn = gencon(num_classes=tlen, lrate=lrate, act=act)  # Create the neural net (a.k.a. "model")
        tb_callback, logdir = KDTFC.gen_tensorboard_callback(nn)  # Saving values for display in a tensorboard

        # Train with validation testing
        nn.fit(inputs, targets, epochs=epochs, batch_size=mbs, validation_split=vf, verbose=2,
        callbacks = [tb_callback])
        KDTFC.fireup_tensorboard(logdir)  # Open a tensorboard in the browser (localhost:6006)

        return nn, logdir


class Critic(
    CriticTable if critic_type == 'table' else (
            CriticANN if critic_type == 'ANN' else False)):
    pass


class RLSystem:
    def __init__(self):
        self.sim_world = SimWorld()
        self.critic = Critic(critic_type, self.sim_world)
        self.actor = Actor(self.critic, self.sim_world)

    def actor_critic_algorithm(self):
        # Initialize
        all_states = self.sim_world.get_all_possible_states()
        self.critic.initialize_value_function(all_states)
        self.actor.initialize_policy_function(all_states)
        acc_reward = [0] * episodes

        for i in range(1, episodes + 1):
            # print("---------- Episode nr: " + str(i) + " ----------")

            # Reset eligibilities in actor and critic
            self.actor.initialize_eligibility_function()
            self.critic.initialize_eligibility_function(all_states)

            # Get S_init and its policy
            state = self.critic.get_state()
            # print("Your start state: " + str(state))
            action = self.actor.get_best_action(state)

            # Play the game
            game_over = self.sim_world.is_game_over()
            while not game_over:
                # Do action
                self.sim_world.do_action(action)
                reward = self.sim_world.get_reward()
                acc_reward[i - 1] += reward
                new_state = self.sim_world.get_state()
                # print("New State: " + str(new_state))

                # Check game status after new state
                game_over = self.sim_world.is_game_over()
                if game_over:
                    break

                # Get new action
                new_action = self.actor.get_best_action(new_state)

                # Update actor's eligibility table
                self.actor.set_eligibility(state, action, 1)

                # Update TD error
                self.critic.set_TD_error(reward, state, new_state)

                # Update critic's eligibility table
                self.critic.set_eligibility(state, 1)

                for s in all_states:
                    self.critic.set_value_for_state(s)
                    self.critic.set_eligibility(s, None)
                    for a in self.actor.policy[s]:
                        self.actor.set_eligibility(s, a, None)
                        self.actor.set_policy(s, a)

                state = new_state
                action = new_action
        self.plot_train_progression(acc_reward)

    def plot_train_progression(self, acc_reward):
        list_episodes = list(range(episodes))
        # Plotting the points
        plt.plot(list_episodes, acc_reward)
        # Plot heller snittet for hvert 100ede episode

        # Name the axis and set title
        plt.xlabel('Episode')
        plt.ylabel('Accumulated reward')
        plt.title('Accumulated reward for each episode')
        plt.show()
