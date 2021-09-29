# callbacks.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

class Callback(object):
    """An object that adds functionality during training.

    A callback is a function or group of functions that can be executed during
    the training process for any of pomegranate's models that have iterative
    training procedures. A callback can be called at three stages-- the
    beginning of training, at the end of each epoch (or iteration), and at
    the end of training. Users can define any functions that they wish in
    the corresponding functions.
    """

    def __init__(self):
        self.model = None
        self.params = None

    def on_training_begin(self):
        """Functionality to add to the beginning of training.

        This method will be called at the beginning of each model's training
        procedure.
        """

        pass

    def on_training_end(self, logs):
        """Functionality to add to the end of training.

        This method will be called at the end of each model's training
        procedure.
        """

        pass

    def on_epoch_end(self, logs):
        """Functionality to add to the end of each epoch.

        This method will be called at the end of each epoch during the model's
        iterative training procedure.
        """

        pass


class History(Callback):
    """Keeps a history of the loss during training."""

    def on_training_begin(self):
        self.total_improvement = []
        self.improvements = []
        self.log_probabilities = []
        self.epoch_start_times = []
        self.epoch_end_times = []
        self.epoch_durations = []
        self.epochs = []
        self.learning_rates = []
        self.n_seen_batches = []
        self.initial_log_probablity = None

    def on_epoch_end(self, logs):
        """Save the files to the appropriate lists."""

        self.total_improvement.append(logs['total_improvement'])
        self.improvements.append(logs['improvement'])
        self.log_probabilities.append(logs['log_probability'])
        self.epoch_start_times.append(logs['epoch_start_time'])
        self.epoch_end_times.append(logs['epoch_end_time'])
        self.epoch_durations.append(logs['duration'])
        self.epochs.append(logs['epoch'])
        self.learning_rates.append(logs['learning_rate'])
        self.n_seen_batches.append(logs['n_seen_batches'])
        self.initial_log_probability = logs['initial_log_probability']
