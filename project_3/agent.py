import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display


class Agent:
    def __init__(self, environment, critic, actor):
        self.environment = environment
        self.critic = critic
        self.actor = actor
        
    
    def prepare_for_epoch(self):
        self.actor.prepare_for_epoch()
        self.critic.prepare_for_epoch()
        discrete_state = self.environment.reset()
        state = self.environment.get_state()
        return state, discrete_state


    def train(self, max_runs=1000, max_length=300):
        print(f'TRAINING STARTED...')
        past_steps = []
        past_loss = []

        # For every Episode...
        for run in range(max_runs):
            episode_loss = []
            state, discrete_state = self.prepare_for_epoch()
            
            # For every step in Episode
            for step in range(max_length):
                
                # Select action and check if it is valid in current state of environment
                action = self.actor.get_action(discrete_state)
                
                # Step with action
                next_state, next_discrete_state, reward, is_state_final = self.environment.step(action, step, run)

                # Compute delta
                if self.critic.nn_critic:
                    delta, loss = self.critic.compute_delta(reward, state, next_state)
                else:
                    delta, loss = self.critic.compute_delta(reward, discrete_state, next_discrete_state)
                    self.critic.trace_eligibilities(delta, discrete_state)
                
                episode_loss.append(loss)

                # Trace eligibilities
                self.actor.trace_eligibilities(delta, discrete_state, action)
                
                # Re-set state
                state, discrete_state = next_state, next_discrete_state

                if is_state_final:
                    break

            past_steps.append(step+1)
            past_loss.append(np.sum(episode_loss))

            if ((run+1) % 100) == 0:
                print(f'Run {run+1}. Step avg: {np.mean(past_steps[-100:]):1.2f}  Loss sum avg.: {np.mean(past_loss[-100:]):1.2f}  Epsilon: {self.actor.expl_rate:1.4f}')
                
                
        print(f'\nGOAL NOT REACHED!\n')
        #self.plot_train_history(past_steps, past_loss)
        
        return past_steps



    def render_episode(self, length=300, sleep=0.01):
        state = self.environment.reset()
        action = np.argmax(self.actor.table[state])
        fig, axs = self.environment.create_figure()
        self.environment.render(fig, axs)
        
        display.display(plt.gcf())
        time.sleep(sleep)
        display.clear_output(wait=True)
        
        for step in range(length):
            action = np.argmax(self.actor.table[state])

            _, state, r, is_state_final = self.environment.step(action, step, 1)

            self.environment.render(fig, axs)
            display.display(plt.gcf())
            time.sleep(sleep)
            display.clear_output(wait=True)

            if is_state_final:
                break