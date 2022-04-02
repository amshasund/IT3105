import tensorflow as tf
from parameters import num_agents, number_actual_games, save_interval


class Tournament:
    def __init__(self) -> None:
        self.agents = []
        
    def create_agents(self):
        list_of_saves = list(range(0, number_actual_games+1, save_interval))
        
        self.agents = []
        for save in list_of_saves:
            self.agents.append(tf.keras.models.load_model("super_model_{}.h5".format(save)))
        
    def create_games():
        pass

    def create_series():
        pass

    def set_up_tournament(self):
        # Create agents with pretrained models
        self.create_agents()
        
        # Create games
        self.create_games()
        
        # Create series
        self.create_series()

    def play_tournament(self):
        # Set up tournament
        self.set_up_tournament()


        

