from plotting import *

# MODEL = "None"
MODEL = "resnet18"
# MODEL = "resnet50"
# MODEL = "nabernet"

TRAIN_PATH = "./logs/train_logger/__" + MODEL + "__run___training.log"
TEST_PATH = "./logs/test_logger/__" + MODEL + "__run___test.log"

pretty_plot(TRAIN_PATH, TEST_PATH, MODEL + "_plot")

