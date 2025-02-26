import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class TweetyNET_Dataset(Dataset):
    def __init__(self, file_dir, num_classes=21, segment_length=370):
        self.file_path = []
        self.num_classes = num_classes
        self.segment_length = segment_length

        for file in os.listdir(file_dir):
            self.file_path.append(os.path.join(file_dir, file))

    def __getitem__(self, index):
        file_path = self.file_path[index]
        data = np.load(file_path, allow_pickle=True)
        spectrogram = data['s']
        ground_truth_labels = data['labels']

        # Z-score normalization
        spectrogram_mean = np.mean(spectrogram)
        spectrogram_std = np.std(spectrogram)
        spectrogram = (spectrogram - spectrogram_mean) / spectrogram_std

        # Convert to tensor
        spectrogram = torch.from_numpy(spectrogram).float().permute(1, 0)  # Bringing spectrogram to length x height
        ground_truth_labels = torch.tensor(ground_truth_labels, dtype=torch.int64).squeeze(0)

        # Convert label to one-hot encoding
        ground_truth_labels = F.one_hot(ground_truth_labels, num_classes=self.num_classes).float()

        return spectrogram, ground_truth_labels 

    def __len__(self):
        return len(self.file_path)

class CollateFunction:
    def __init__(self, segment_length=370):
        self.segment_length = segment_length

    def __call__(self, batch):
        # Unzip the batch (a list of (spectrogram, ground_truth_labels) tuples)
        spectrograms, ground_truth_labels = zip(*batch)

        # Create lists to hold the processed tensors
        spectrograms_processed = []
        ground_truth_labels_processed = []

        # Each sample in batch
        for spectrogram, ground_truth_label in zip(spectrograms, ground_truth_labels):
            # Truncate if larger than context window or take random window
            if spectrogram.shape[0] > self.segment_length:
                start = torch.randint(0, spectrogram.shape[0] - self.segment_length, (1,)).item()
                end = start + self.segment_length
                spectrogram = spectrogram[start:end]
                ground_truth_label = ground_truth_label[start:end]
            elif spectrogram.shape[0] < self.segment_length:
                # Pad with 0s if shorter
                pad_amount = self.segment_length - spectrogram.shape[0]
                spectrogram = F.pad(spectrogram, (0, 0, 0, pad_amount), 'constant', 0)
                ground_truth_label = F.pad(ground_truth_label, (0, 0, 0, pad_amount), 'constant', 0)

            # Append the processed tensors to the lists
            spectrograms_processed.append(spectrogram)
            ground_truth_labels_processed.append(ground_truth_label)

        # Stack tensors along a new dimension
        spectrograms = torch.stack(spectrograms_processed, dim=0)
        ground_truth_labels = torch.stack(ground_truth_labels_processed, dim=0)

        # Final reshape for model
        spectrograms = spectrograms.unsqueeze(1).permute(0, 1, 3, 2)

        return spectrograms, ground_truth_labels


class Conv2dTF(nn.Conv2d):

    PADDING_METHODS = ('VALID', 'SAME')

    """Conv2d with padding behavior from Tensorflow
    adapted from
    https://github.com/mlperf/inference/blob/16a5661eea8f0545e04c86029362e22113c2ec09/others/edge/object_detection/ssd_mobilenet/pytorch/utils.py#L40
    as referenced in this issue:
    https://github.com/pytorch/pytorch/issues/3867#issuecomment-507025011
    used to maintain behavior of original implementation of TweetyNet that used Tensorflow 1.0 low-level API
    """
    def __init__(self, *args, **kwargs):
        # remove 'padding' from ``kwargs`` to avoid bug in ``torch`` => 1.7.2
        # see https://github.com/yardencsGitHub/tweetynet/issues/166
        kwargs_super = {k: v for k, v in kwargs.items() if k != 'padding'}
        super(Conv2dTF, self).__init__(*args, **kwargs_super)
        padding = kwargs.get("padding", "SAME")
        if not isinstance(padding, str):
            raise TypeError(f"value for 'padding' argument should be a string, one of: {self.PADDING_METHODS}")
        padding = padding.upper()
        if padding not in self.PADDING_METHODS:
            raise ValueError(
                f"value for 'padding' argument must be one of '{self.PADDING_METHODS}' but was: {padding}"
            )
        self.padding = padding

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif self.padding == "SAME":
            rows_odd, padding_rows = self._compute_padding(input, dim=0)
            cols_odd, padding_cols = self._compute_padding(input, dim=1)
            if rows_odd or cols_odd:
                input = F.pad(input, [0, cols_odd, 0, rows_odd])

            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=(padding_rows // 2, padding_cols // 2),
                dilation=self.dilation,
                groups=self.groups,
            )

class TweetyNet(nn.Module):
    def __init__(self,
                 num_classes,
                 input_shape=(1, 512, 370),
                 padding='SAME',
                 conv1_filters=32,
                 conv1_kernel_size=(5, 5),
                 conv2_filters=64,
                 conv2_kernel_size=(5, 5),
                 pool1_size=(14, 1),
                 pool1_stride=(14, 1),
                 pool2_size=(14, 1),
                 pool2_stride=(14, 1),
                 hidden_size=512,
                 rnn_dropout=0.,
                 num_layers=1,
                 bidirectional=True,
                 ):
        """initialize TweetyNet model
        Parameters
        ----------
        num_classes : int
            number of classes to predict, e.g., number of syllable classes in an individual bird's song
        input_shape : tuple
            with 3 elements corresponding to dimensions of spectrogram windows: (channels, frequency bins, time bins).
            i.e. we assume input is a spectrogram and treat it like an image, typically with one channel,
            the rows are frequency bins, and the columns are time bins. Default is (1, 513, 88).
        padding : str
            type of padding to use, one of {"VALID", "SAME"}. Default is "SAME".
        conv1_filters : int
            Number of filters in first convolutional layer. Default is 32.
        conv1_kernel_size : tuple
            Size of kernels, i.e. filters, in first convolutional layer. Default is (5, 5).
        conv2_filters : int
            Number of filters in second convolutional layer. Default is 64.
        conv2_kernel_size : tuple
            Size of kernels, i.e. filters, in second convolutional layer. Default is (5, 5).
        pool1_size : two element tuple of ints    specs = self.transform(specs)

            Size of sliding window for first max pooling layer. Default is (1, 8)
        pool1_stride : two element tuple of ints
            Step size for sliding window of first max pooling layer. Default is (1, 8)
        pool2_size : two element tuple of ints
            Size of sliding window for second max pooling layer. Default is (1, 8),
        pool2_stride : two element tuple of ints
            Step size for sliding window of second max pooling layer. Default is (1, 8)
        hidden_size : int
            number of features in the hidden state ``h``. Default is None,
            in which case ``hidden_size`` is set to the dimensionality of the
            output of the convolutional neural network. This default maintains
            the original behavior of the network.
        rnn_dropout : float
            If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
            with dropout probability equal to dropout. Default: 0
        num_layers : int
            Number of recurrent layers. Default is 1.
        bidirectional : bool
            If True, make LSTM bidirectional. Default is True.
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape

        self.cnn = nn.Sequential(
            Conv2dTF(in_channels=self.input_shape[0],
                     out_channels=conv1_filters,
                     kernel_size=conv1_kernel_size,
                     padding=padding
                     ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool1_size,
                         stride=pool1_stride),
            Conv2dTF(in_channels=conv1_filters,
                     out_channels=conv2_filters,
                     kernel_size=conv2_kernel_size,
                     padding=padding,
                     ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool2_size,
                         stride=pool2_stride),
        )

        # determine number of features in output after stacking channels
        # we use the same number of features for hidden states
        # note self.num_hidden is also used to reshape output of cnn in self.forward method
        batch_shape = tuple((1,) + input_shape)
        tmp_tensor = torch.rand(batch_shape)
        tmp_out = self.cnn(tmp_tensor)
        channels_out, freqbins_out = tmp_out.shape[1], tmp_out.shape[2]
        self.rnn_input_size = channels_out * freqbins_out

        if hidden_size is None:
            self.hidden_size = self.rnn_input_size
        else:
            self.hidden_size = hidden_size

        self.rnn = nn.LSTM(input_size=self.rnn_input_size,
                           hidden_size=self.hidden_size,
                           num_layers=num_layers,
                           dropout=rnn_dropout,
                           bidirectional=bidirectional)

        # for self.fc, in_features = hidden_size * 2 because LSTM is bidirectional
        # so we get hidden forward + hidden backward as output
        self.fc = nn.Linear(in_features=self.hidden_size * 2, out_features=num_classes)

    def forward(self, x):
        features = self.cnn(x)
        # stack channels, to give tensor shape (batch, rnn_input_size, num time bins)
        features = features.view(features.shape[0], self.rnn_input_size, -1)
        # switch dimensions for feeding to rnn, to (num time bins, batch size, input size)
        features = features.permute(2, 0, 1)
        rnn_output, _ = self.rnn(features)
        # permute back to (batch, time bins, hidden size) to project features down onto number of classes
        rnn_output = rnn_output.permute(1, 0, 2)
        logits = self.fc(rnn_output)
        # permute yet again so that dimension order is (batch, classes, time steps)
        # because this is order that loss function expects
        return logits.permute(0, 2, 1)

    def loss_function(self, y_pred, y_true):
        """loss function for TweetyNet
        Parameters
        ----------
        y_pred : torch.Tensor
            output of TweetyNet model, shape (batch, classes, timebins)
        y_true : torch.Tensor
            one-hot encoded labels, shape (batch, classes, timebins)
        Returns
        -------
        loss : torch.Tensor
            mean cross entropy loss
        """
        loss = nn.CrossEntropyLoss()
        return loss(y_pred, y_true)


def frame_error_rate(y_pred, y_true):
    """
    Compute the frame error rate.
    y_pred: Tensor of shape (batch_size, num_classes, time_steps) - typically the output of a softmax
    y_true: Tensor of shape (batch_size, time_steps) - ground truth labels
    Returns the frame error rate.
    """
    predicted_labels = y_pred.argmax(dim=-2)
    mismatches = (predicted_labels != y_true).float()  # y_true is already in the correct format
    error = mismatches.sum() / y_true.size(0) / y_true.size(1)
    return error * 100

class TweetyNetTrainer:
    def __init__(self, model, train_loader, test_loader, device, optimizer, desired_total_steps=1e4, patience=4, batches_per_eval=50, plotting=False, moving_avg_window=1):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = optimizer
        self.desired_total_steps = desired_total_steps
        self.patience = patience
        self.batches_per_eval = batches_per_eval
        self.total_steps = 0
        self.best_val_loss = float('inf')
        self.stop_training = False
        self.loss_list = []
        self.val_loss_list = []
        self.frame_error_rate_list = []
        self.plotting = plotting
        self.moving_avg_window = moving_avg_window

    def moving_average(self, values, window):
        """Simple moving average over a list of values"""
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, 'valid')
        return sma.tolist()

    def validate_model(self, test_iter):
        self.model.eval()
        total_val_loss = 0
        total_frame_error = 0
        with torch.no_grad():
            try:
                spectrogram, label = next(test_iter)
            except StopIteration:
                test_iter = iter(self.test_loader)  # Reinitialize the iterator
                spectrogram, label = next(test_iter)
            spectrogram = spectrogram.to(self.device)
            label = label.to(self.device)
            output = self.model(spectrogram)  # Use the model to predict
            label = label.squeeze(1)
            label_indices = label.argmax(dim=-1)
            loss = self.model.loss_function(y_pred=output, y_true=label_indices)

            total_val_loss = loss.item()
            total_frame_error = frame_error_rate(output, label_indices).item()

        return total_val_loss, total_frame_error

    def train(self):
        best_val_loss = float('inf')
        num_val_no_improve = 0

        raw_loss_list, raw_val_loss_list, raw_frame_error_rate_list = [], [], []
        moving_avg_val_loss_list, moving_avg_frame_error_list = [], []

        train_iter = iter(self.train_loader)
        test_iter = iter(self.test_loader)

        while self.total_steps < self.desired_total_steps and not self.stop_training:
            try:
                spectrogram, label = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)  # Reinitialize the iterator
                spectrogram, label = next(train_iter)

            self.model.train()
            spectrogram = spectrogram.to(self.device)
            label = label.to(self.device)
            output = self.model(spectrogram)  # Use the model to predict
            label = label.squeeze(1)
            label = label.permute(0, 2, 1)
            label_indices = label.argmax(dim=-2)
            loss = self.model.loss_function(y_pred=output, y_true=label_indices)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.total_steps += 1

            if self.total_steps % self.batches_per_eval == 0:
                val_loss, frame_error = self.validate_model(test_iter)

                raw_loss_list.append(loss.item())
                raw_val_loss_list.append(val_loss)
                raw_frame_error_rate_list.append(frame_error)

                if len(raw_val_loss_list) >= self.moving_avg_window:
                    moving_avg_val_loss = np.mean(raw_val_loss_list[-self.moving_avg_window:])
                    moving_avg_frame_error = np.mean(raw_frame_error_rate_list[-self.moving_avg_window:])
                    moving_avg_val_loss_list.append(moving_avg_val_loss)
                    moving_avg_frame_error_list.append(moving_avg_frame_error)

                    if moving_avg_val_loss < best_val_loss:
                        best_val_loss = moving_avg_val_loss
                        num_val_no_improve = 0
                    else:
                        num_val_no_improve += 1
                        if num_val_no_improve >= self.patience:
                            print("Early stopping triggered")
                            self.stop_training = True
                            break
                
                print(f'Step {self.total_steps}: Train Loss {loss.item():.4f} FER = {frame_error:.2f}%, Val Loss = {val_loss:.4f}')

            if self.stop_training:
                break

            if self.plotting:
                self.plot_results(raw_loss_list, moving_avg_val_loss_list, moving_avg_frame_error_list)


    def plot_results(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_list, label='Smoothed Training Loss')
        plt.plot(self.val_loss_list, label='Smoothed Validation Loss')
        plt.title('Smoothed Loss over Steps')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.frame_error_rate_list, label='Smoothed Frame Error Rate', color='red')
        plt.title('Smoothed Frame Error Rate over Steps')
        plt.xlabel('Steps')
        plt.ylabel('Error Rate (%)')
        plt.legend()

        plt.tight_layout()
        plt.show()

        print(f"Final smoothed loss after {self.total_steps} batches: {self.loss_list[-1]}")


class ModelEvaluator:
    def __init__(self, model, test_loader, num_classes=21, device='cuda:0', filter_unseen_classes=False, train_dir=None):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.device = device
        self.filter_unseen_classes = filter_unseen_classes
        self.seen_classes = set(range(num_classes))  # By default, consider all classes as seen
        if filter_unseen_classes and train_dir:
            self.seen_classes = self.count_labels_in_training_set(train_dir)

    def count_labels_in_training_set(self, train_dir):
        seen_classes = set()
        for file in os.listdir(train_dir):
            if file.endswith('.npz'):
                data = np.load(os.path.join(train_dir, file))
                labels = data['labels']
                unique_labels = set(np.unique(labels))
                seen_classes.update(unique_labels)
        return seen_classes

    def validate_model_multiple_passes(self, num_passes=1, max_batches=10000):
        self.model.eval()
        errors_per_class = [0] * self.num_classes
        correct_per_class = [0] * self.num_classes
        total_frames = 0
        total_errors = 0

        total_iterations = num_passes * min(max_batches, len(self.test_loader))
        progress_bar = tqdm(total=total_iterations, desc="Evaluating", unit="batch")

        for _ in range(num_passes):
            with torch.no_grad():
                for i, (waveform, label) in enumerate(self.test_loader):
                    if i >= max_batches:
                        break

                    waveform = waveform.to(self.device)
                    label = label.to(self.device)

                    output = self.model.forward(waveform)
                    label = label.squeeze(1).permute(0, 2, 1)

                    predicted_labels = output.argmax(dim=-2)
                    true_labels = label.argmax(dim=-2)

                    correct = (predicted_labels == true_labels)
                    incorrect = ~correct

                    for cls in range(self.num_classes):
                        if self.filter_unseen_classes and cls not in self.seen_classes:
                            continue  # Skip unseen classes

                        class_mask = true_labels == cls
                        incorrect_class = incorrect & class_mask

                        errors_per_class[cls] += incorrect_class.sum().item()
                        correct_per_class[cls] += (correct_class := correct & class_mask).sum().item()

                        total_frames += class_mask.sum().item()
                        total_errors += incorrect_class.sum().item()

                    progress_bar.update(1)

        progress_bar.close()

        class_frame_error_rates = {
            cls: (errors / (errors + correct) * 100 if errors + correct > 0 else float('nan'))
            for cls, (errors, correct) in enumerate(zip(errors_per_class, correct_per_class))
        }

        total_frame_error_rate = (total_errors / total_frames * 100 if total_frames > 0 else float('nan'))

        return class_frame_error_rates, total_frame_error_rate

    def plot_error_rates(self, class_frame_error_rates, save_path):
        classes = [cls for cls in range(self.num_classes) if cls in self.seen_classes or not self.filter_unseen_classes]
        error_rates = [class_frame_error_rates[cls] for cls in classes]

        plt.figure(figsize=(10, 6))
        plt.bar(classes, error_rates, color='skyblue')
        plt.xlabel('Class')
        plt.ylabel('Error Rate (%)')
        plt.title('Class-wise Error Rates')
        plt.xticks(classes)
        plt.ylim(0, max(error_rates) + 5)  # Adjust Y-axis to show the highest error rate clearly
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'class_error_rates.png'))
        plt.close()

    def save_results(self, class_frame_error_rates, total_frame_error_rate, folder_path):
        results_data = {
            'class_frame_error_rates': class_frame_error_rates,
            'total_frame_error_rate': total_frame_error_rate
        }

        json_filename = 'class_frame_error_rates.json'
        with open(os.path.join(folder_path, json_filename), 'w') as file:
            json.dump(results_data, file)

