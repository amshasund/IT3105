import tensorflow as tf
from parameters import num_agents, number_actual_games, save_interval


class Tournament:
    def __init__(self) -> None:
        pass

    def create_agents(self):
        list_of_saves = list(range(save_interval, number_actual_games+1, save_interval))
        print(list_of_saves)
        pass
        for save in list_of_saves:
            reconstructed_model = tf.keras.models.load_model("super_model_{}.h5".format(save))

    def set_up_tournament(self):
        pass

    def play_tournament(self):
        pass

