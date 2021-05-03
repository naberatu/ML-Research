from plot_routine import *


class Plot:
    def __init__(self, model_name):
        self.model_name = model_name
        self.TRAIN_PATH = "./logs/train_logger/__" + self.model_name + "__run___training.log"
        self.TEST_PATH = "./logs/test_logger/__" + self.model_name + "__run___test.log"
        # self.TEST2_PATH = "./logs/test_logger/--" + self.model_name + "__split___test.log"

    def plot(self):
        pretty_plot(self.TRAIN_PATH, self.TEST_PATH, self.model_name + "_plot")
        # pretty_plot(self.TRAIN_PATH, self.TEST2_PATH, self.model_name + "_plot2")


if __name__ == '__main__':
    Plot("nabernet_c2").plot()
    # If you're gonna retest a model, erase its log first.

