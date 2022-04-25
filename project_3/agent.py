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
                
                action = self.actor.get_action(discrete_state)
                
                next_state, next_discrete_state, reward, is_state_final = self.environment.step(action, step, run)

                delta, loss = self.critic.compute_delta(reward, state, next_state)
                
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
                
        self.plot_train_history(past_steps, past_loss)
                
        return past_steps
    
    
    def plot_train_history(self, past_steps, past_loss=[]):
        fig, ax = plt.subplots(2, figsize=(16, 8))
        ax[0].set_ylim([0, np.max(past_steps)+20])
        ax[0].plot(past_steps)
        if(past_loss != []):
            ax[1].set_ylim([0, np.median(past_loss)*4])
            ax[1].plot(past_loss)



    def render_episode(self, length=500):
        state = self.environment.reset()
        fig, ax = plt.subplots(1, figsize=(12, 12))
        
        for step in range(length):
            action = np.argmax(self.actor.table[state])

            _, state, r, is_state_final = self.environment.step(action, step, 1)

            self.environment.render(fig, ax, action)
            display.display(plt.gcf())
            display.clear_output(wait=True)

            if is_state_final:
                break