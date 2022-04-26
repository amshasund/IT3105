import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display


class Agent:
    
    N = 4
    
    def __init__(self, environment, critic, actor):
        self.environment = environment
        self.critic = critic
        self.actor = actor
        
    
    def prepare_for_epoch(self):
        self.actor.prepare_for_epoch()
        discrete_state = self.environment.reset()
        state = self.environment.get_state()
        return state, discrete_state


    def train(self, max_runs, max_length):
        print(f'TRAINING STARTED...')
        past_steps, past_loss = [], []

        for run in range(max_runs):
            episode_loss = []
            state, discrete_state = self.prepare_for_epoch()
            
            for step in range(max_length):
                
                if step%self.N == 0:
                    action = self.actor.get_action(discrete_state)  # Get action from actor
                
                next_state, next_discrete_state, reward, is_state_final = self.environment.step(action, step)  # Perform action in environment

               
                delta, loss = self.critic.compute_delta(reward, state, next_state)  # Compute delta in critic
                episode_loss.append(loss)

                self.actor.trace_eligibilities(delta, discrete_state, action)  # Trace eligibilities in actor
                
                state, discrete_state = next_state, next_discrete_state

                if is_state_final:
                    break

            past_steps.append(step+1)
            past_loss.append(np.sum(episode_loss))
            
            if step + 1 <= 400:
                print(f'Run {run}: GOAL STEPS <= 400: {step+1}')

            if ((run+1) % 100) == 0:
                print(f'Run {run+1}. Step avg: {np.mean(past_steps[-100:]):1.2f}  Loss sum avg.: {np.mean(past_loss[-100:]):1.2f}  Epsilon: {self.actor.expl_rate:1.4f}')
            
            if self.environment.is_goal_reached(past_steps):
                print(f'\n\nGOAL REACHED!!!\n\n')
                break
            
        self.plot_train_history(past_steps, past_loss)
                
        return past_steps
    
    
    
    def plot_train_history(self, past_steps, past_loss=[]):
        fig, ax = plt.subplots(2, figsize=(16, 8))
        ax[0].set_ylim([0, np.max(past_steps)+20])
        ax[0].plot(past_steps)
        ax[0].plot(past_steps)
        ax[0].axhline(min(past_steps), ls='--', color='black')
        ax[0].text(0, 1, f'Min Steps: {min(past_steps)}', ha='left', va='bottom', size=10)
        if(past_loss != []):
            ax[1].set_ylim([0, np.median(past_loss)*4])
            ax[1].plot(past_loss)



    def render_episode(self, length=500):
        num_actions = 0
        
        state = self.environment.reset()
        fig, ax = plt.subplots(1, figsize=(12, 12))
        
        for step in range(length):
            
            if step % self.N == 0:
                action = np.argmax(self.actor.table[state])
                num_actions += 1
            
            _, state, r, is_state_final = self.environment.step(action, step)

            self.environment.render(fig, ax, action, step, num_actions)
            display.display(plt.gcf())
            display.clear_output(wait=True)

            if is_state_final:
                break