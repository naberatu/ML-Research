import matplotlib.pyplot as plt
from collections import defaultdict
import re
import pathlib


def creating_path(folder_name: str, file_name: str, extension: str):
    """
    This function is taken folder name, and file with desired extension for the file
    the goal to have poxis path in order to write or read from it
    it is more likily to use the function inside the code not for other things
    :return path_file
    """
    path = str(pathlib.Path.cwd()) + '/' + folder_name
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    file_path = str(path) + "/" + file_name + "." + extension

    return file_path


def parse_test(path_log_file):
    """
    Mohammed Alneamri
    the idea of this function is to prpaer for ploting
    so we have to plot the log file we generated using logging
    so it is a little bit of hacking I have to change little bit
    maybe using the index or rindex to have the epoch, loss , acc1, acc5
    using defaultdict becuae of having dict in dict

    :param path_log_file: path of testing log file
    :return: list epochs, loss, acc1,acc5
    """
    loss = defaultdict(list)
    acc1 = defaultdict(list)
    # acc5 = defaultdict(list)
    with open(path_log_file) as file:
        for line in file:
            items = line.split("\t")
            epoch = re.sub("[^0-9]", "", items[0][8:11])
            loss[int(epoch)].append(float(re.sub("[^0-9,.]", "",items[1][13:19])))
            acc1[int(epoch)].append(float(re.sub("[^0-9,.]", "",items[2][15:21])))
            # acc5[int(epoch)].append(float(re.sub("[^0-9,.]", "",items[3][15:21])))
    loss_dict = {}
    acc1_dict = {}
    # acc5_dict = {}
    for epoch, loss in loss.items():
        loss_dict[epoch] = round(sum(loss) / len(loss), 3)
    for epoch, acc1 in acc1.items():
        acc1_dict[epoch] = round(sum(acc1) / len(acc1), 3)
    # for epoch, acc5 in acc5.items():
    #     acc5_dict[epoch] = round(sum(acc5) / len(acc5), 3)

    epochs = list(acc1_dict.keys())
    loss = list(loss_dict.values())
    acc1 = list(acc1_dict.values())
    # acc5 = list(acc5_dict.values())

    # return epochs, loss, acc1, acc5
    return epochs, loss, acc1


def parse_train(path_log_file):
    """
    Mohammed Alneamri
    the idea of this function is to prpaer for ploting
    so we have to plot the log file we generated using logging
    so it is a little bit of hacking I have to change little bit
    maybe using the index or rindex to have the epoch, loss , acc1, acc5
    using defaultdict becuae of having dict in dict
    the only difference here the log has data, time whcih has the index 1,2
    :param path_log_file: path of testing log file
    :return: list epochs, loss, acc1,acc5
    """

    loss = defaultdict(list)
    acc1 = defaultdict(list)
    # acc5 = defaultdict(list)
    with open(path_log_file) as file:
        for line in file:
            items = line.split("\t")
            epoch = re.sub("[^0-9]", "", items[0][8:11])
            loss[int(epoch)].append(float(re.sub("[^0-9,.]", "",items[3][13:19])))
            acc1[int(epoch)].append(float(re.sub("[^0-9,.]", "",items[4][15:21])))
            # acc5[int(epoch)].append(float(re.sub("[^0-9,.]", "",items[5][15:21])))
    loss_dict = {}
    acc1_dict = {}
    # acc5_dict = {}
    for epoch, loss in loss.items():
        loss_dict[epoch] = round(sum(loss) / len(loss), 3)
    for epoch, acc1 in acc1.items():
        acc1_dict[epoch] = round(sum(acc1) / len(acc1), 3)
    # for epoch, acc5 in acc5.items():
    #     acc5_dict[epoch] = round(sum(acc5) / len(acc5), 3)

    epochs = list(acc1_dict.keys())
    loss = list(loss_dict.values())
    acc1 = list(acc1_dict.values())
    # acc5 = list(acc5_dict.values())

    # return epochs, loss, acc1, acc5
    return epochs, loss, acc1


def pretty_plot(path_train_log, path_test_log, name_model="Model"):
    """
    Written by Mohammed

    this is will plot the figures for you with accuracy anmd on the graph
    very nice and bea
    todo this function is not general can only paint four lines 2 for train and 2 fro test for one model
    todo I Need to make work with list of the patth and can plot differtrnt model in the same figures
    """

    fig, ax = plt.subplots()
    # epochs, loss_train, acc1_train, acc5_train = parse_train(path_train_log)
    # epochs, loss_test, acc1_test, acc5_test = parse_test(path_test_log)
    epochs, loss_train, acc1_train = parse_train(path_train_log)
    epochs, loss_test, acc1_test = parse_test(path_test_log)

    acc1_p_tr, = ax.plot(epochs, acc1_train, label='acc1_train', linestyle='--', color='g', marker='D', markersize=5,
                         linewidth=2)
    # acc5_p_tr, = ax.plot(epochs, acc5_train, label='acc5_train', linestyle='--', color='b', marker='D', markersize=5,
    #                      linewidth=2)

    acc1_p_ts, = ax.plot(epochs, acc1_test, label='acc1_test', linestyle='--', color='r', marker='D', markersize=5,
                         linewidth=2)
    # acc5_p_ts, = ax.plot(epochs, acc5_test, label='acc5_test', linestyle='--', color='c', marker='D', markersize=5,
    #                      linewidth=2)

    ax.set_title(name_model)
    # ax.se
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epochs')
    ax.axis('tight')
    plt.annotate('Train Acc1: {}'.format(max(acc1_train)),
                 xy=(acc1_train.index(max(acc1_train)), max(acc1_train)), xycoords='data',
                 xytext=(+10, +15), textcoords='offset points', fontsize=10,
                 bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec=(None)),
                 arrowprops=dict(arrowstyle="wedge,tail_width=1.",
                                 fc=(1.0, 0.7, 0.7), ec=(1., .5, .5),
                                 patchA=None,
                                 patchB=None,
                                 relpos=(0.2, 0.8),
                                 connectionstyle="arc3,rad=-0.1"))

    # plt.annotate('Train Acc5: {}'.format(max(acc5_train)),
    #              xy=(acc5_train.index(max(acc5_train)), max(acc5_train)), xycoords='data',
    #              xytext=(-37, -30), textcoords='offset points', fontsize=10,
    #              bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec=(None)),
    #              arrowprops=dict(arrowstyle="wedge,tail_width=1.",
    #                              fc=(1.0, 0.7, 0.7), ec=(1., .5, .5),
    #                              patchA=None,
    #                              patchB=None,
    #                              relpos=(0.2, 0.8),
    #                              connectionstyle="arc3,rad=-0.1"))

    plt.annotate('Test Acc1: {}'.format(max(acc1_test)),
                 xy=(acc1_test.index(max(acc1_test)), max(acc1_test)), xycoords='data',
                 xytext=(+10, +15), textcoords='offset points', fontsize=10,
                 bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec=(None)),
                 arrowprops=dict(arrowstyle="wedge,tail_width=1.",
                                 fc=(1.0, 0.7, 0.7), ec=(1., .5, .5),
                                 patchA=None,
                                 patchB=None,
                                 relpos=(0.2, 0.8),
                                 connectionstyle="arc3,rad=-0.1"))

    # plt.annotate('Test Acc5: {}'.format(max(acc5_test)),
    #              xy=(acc5_test.index(max(acc5_test)), max(acc5_test)), xycoords='data',
    #              xytext=(+10, -15), textcoords='offset points', fontsize=10,
    #              bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec=(None)),
    #              arrowprops=dict(arrowstyle="wedge,tail_width=1.",
    #                              fc=(1.0, 0.7, 0.7), ec=(1., .5, .5),
    #                              patchA=None,
    #                              patchB=None,
    #                              relpos=(0.2, 0.8),
    #                              connectionstyle="arc3,rad=-0.1"))

    # ax.legend(handles=[acc1_p_tr, acc5_p_tr, acc1_p_ts, acc5_p_ts])
    ax.legend(handles=[acc1_p_tr, acc1_p_ts])
    ax.grid()
    figurename = creating_path("figures", name_model, "png")
    plt.savefig(figurename, dpi=300)
    plt.show()

