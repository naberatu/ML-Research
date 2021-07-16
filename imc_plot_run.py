from imc_plot_routine import *


class Plot:
    def __init__(self, model_name):
        self.model_name = model_name
        self.TRAIN_PATH = "./logs/train_logger/__" + self.model_name + "__run___training.log"
        self.TEST_PATH = "./logs/test_logger/__" + self.model_name + "__run___test.log"

    def plot(self, only_test=False):
        if only_test:
            self.TEST_PATH = "./logs/test_logger/__" + self.model_name + "_eval.log"
            pretty_plot(self.TRAIN_PATH, self.TEST_PATH, self.model_name + "_eval_plot", test_only=only_test)
        else:
            pretty_plot(self.TRAIN_PATH, self.TEST_PATH, self.model_name + "_plot", test_only=only_test)


if __name__ == '__main__':
    name = input("Input Model Name:\t\t")
    test = input("Testing Only (y/n)?\t\t")
    Plot(name).plot(only_test='y' in test.lower())

