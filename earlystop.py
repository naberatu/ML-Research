from collections import deque


class EarlyStopping(object):
    def __init__(self, patience=8):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.previous_loss = int(1e8)
        self.previous_accuracy = 0
        self.init = False
        self.accuracy_decrease_iters = 0
        self.loss_increase_iters = 0
        self.best_running_accuracy = 0
        self.best_running_loss = int(1e7)

    def add_data(self, model, loss, accuracy):

        # compute moving average
        if not self.init:
            running_loss = loss
            running_accuracy = accuracy
            self.init = True

        else:
            running_loss = 0.2 * loss + 0.8 * self.previous_loss
            running_accuracy = 0.2 * accuracy + 0.8 * self.previous_accuracy

        # check if running accuracy has improved beyond the best running accuracy recorded so far
        if running_accuracy < self.best_running_accuracy:
            self.accuracy_decrease_iters += 1
        else:
            self.best_running_accuracy = running_accuracy
            self.accuracy_decrease_iters = 0

        # check if the running loss has decreased from the best running loss recorded so far
        if running_loss > self.best_running_loss:
            self.loss_increase_iters += 1
        else:
            self.best_running_loss = running_loss
            self.loss_increase_iters = 0

        # log the current accuracy and loss
        self.previous_accuracy = running_accuracy
        self.previous_loss = running_loss

    def stop(self):

        # compute thresholds
        accuracy_threshold = self.accuracy_decrease_iters > self.patience
        loss_threshold = self.loss_increase_iters > self.patience

        # return codes corresponding to exhuaustion of patience for either accuracy or loss
        # or both of them
        if accuracy_threshold and loss_threshold:
            return 1

        if accuracy_threshold:
            return 2

        if loss_threshold:
            return 3

        return 0

    def reset(self):
        # reset
        self.accuracy_decrease_iters = 0
        self.loss_increase_iters = 0


