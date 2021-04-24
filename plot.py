from plot_routine import *


class Plot:
    def __init__(self, model_name):
        self.model_name = model_name
        self.TRAIN_PATH = "./logs/train_logger/__" + self.model_name + "__run___training.log"
        self.TEST_PATH = "./logs/test_logger/__" + self.model_name + "__run___test.log"

    def plot(self):
        pretty_plot(self.TRAIN_PATH, self.TEST_PATH, self.model_name + "_plot")


if __name__ == '__main__':
    Plot("nabernet_b").plot()
