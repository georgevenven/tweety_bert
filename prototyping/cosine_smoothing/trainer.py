import os
import numpy as np 
import torch
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import json
from torch.profiler import profile, record_function, ProfilerActivity
from itertools import cycle
from torch.cuda.amp import autocast, GradScaler

class ModelTrainer:
    def __init__(self, model, train_loader, test_loader, optimizer, device, 
             max_steps=10000, eval_interval=500, save_interval=1000, 
             weights_save_dir='saved_weights', overfit_on_batch=False, experiment_dir=None, early_stopping=True, patience=8, trailing_avg_window=1000):

        self.overfit_on_batch = overfit_on_batch
        self.fixed_batch = None  # Will hold the batch data when overfitting
        self.model = model
        self.train_iter = train_loader
        self.test_iter = test_loader
        self.optimizer = optimizer
        self.device = device
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.scheduler = StepLR(optimizer, step_size=10000, gamma=1)
        self.early_stopping = early_stopping
        self.patience = patience 
        self.trailing_avg_window = trailing_avg_window  # Window size for trailing average calculation

        self.save_interval = save_interval
        self.weights_save_dir = weights_save_dir
        self.experiment_dir = experiment_dir  # Assuming the experiment dir is the parent of visualizations_save_dir

        self.count_reinit=0
        
        # Create directories if they do not exist
        if not os.path.exists(self.weights_save_dir):
            os.makedirs(self.weights_save_dir)

        # Create a subfolder for predictions under visualizations_save_dir
        self.predictions_subfolder_path = os.path.join(experiment_dir, "predictions")
        if not os.path.exists(self.predictions_subfolder_path):
            os.makedirs(self.predictions_subfolder_path)

    def sum_squared_weights(self):
        sum_of_squares = sum(torch.sum(p ** 2) for p in self.model.parameters())
        return sum_of_squares   
 
    def save_model(self, step, training_stats):
        # Save model weights with the step number to maintain a history of models
        filename = f"model_step_{step}.pth"
        filepath = os.path.join(self.weights_save_dir, filename)
        torch.save(self.model.state_dict(), filepath)
        
        # Save training statistics in a fixed file, overwriting previous content
        stats_filename = "training_statistics.json"
        stats_filepath = os.path.join(self.experiment_dir, stats_filename)
        with open(stats_filepath, 'w') as json_file:
            json.dump(training_stats, json_file, indent=4)

    def create_large_canvas(self, intermediate_outputs, image_idx=0, spec_shape=None):
        """
        Create a large canvas that places all items in the dictionary.
        Make sure the x-axis is aligned for all of them.
        """
        num_layers = len(intermediate_outputs)
        fig, axes = plt.subplots(num_layers, 1, figsize=(10, 3 * num_layers))  # All images in one column

        # Determine the y-axis limit based on the 'spec' shape if provided
        y_axis_limit = spec_shape[1] if spec_shape else None

        for ax, (name, tensor) in zip(axes, intermediate_outputs.items()):
            # Convert PyTorch tensor to NumPy array
            image_data = tensor[image_idx].cpu().detach().numpy()

            # Remove singleton dimensions
            image_data = np.squeeze(image_data)

            # If the tensor is 1D, reshape it to 2D
            if len(image_data.shape) == 1:
                image_data = np.expand_dims(image_data, axis=0)

            # Now, make sure it's 2D before plotting
            if len(image_data.shape) == 2:
                ax.imshow(image_data)  # Adjust based on your actual data
                if y_axis_limit:
                    ax.set_ylim(bottom=[0, y_axis_limit])  # Set the y-axis limit to match the 'spec'
                ax.set_aspect('auto')  # This will ensure that the y-axis size is the same in pixels for all plots
                ax.invert_yaxis()  # Invert the y-axis
            else:
                print(f"Skipping {name}, as it is not 1D or 2D after squeezing. Shape is {image_data.shape}")

            ax.set_title(name)
            ax.axis('off')

        plt.tight_layout()

    def visualize_mse(self, output, mask, spec, step):
        mask_bar_height = 15
        output = output.cpu().numpy()  # Assuming output has shape [1, seq_len, 196]

        # Process the inputs
        spec_np = spec.squeeze(1).cpu().numpy()
        mask_np = mask[:, 0, :].cpu().numpy()

        # Prepare plots with 2 rows instead of 3
        fig, axs = plt.subplots(2, 1, figsize=(30, 20))  # Adjusted figsize for 2 plots
        axs = axs.ravel()

        # Labels for X and Y axes
        x_label = 'Time Bins'
        y_label = 'Frequency Bins'

        # Adjust spacing between figures, and between titles and figures
        plt.subplots_adjust(hspace=0.33)  # Adjust vertical space between plots

        for ax in axs:
            ax.tick_params(axis='both', which='major', labelsize=25, length=6, width=2)  # Adjust tick length and width
            ax.set_xlabel(x_label, fontsize=25)
            ax.set_ylabel(y_label, fontsize=25)

        # Plot 1: Original Spectrogram with Mask
        axs[0].imshow(spec_np[0], aspect='auto', origin='lower')
        axs[0].set_title('Original Spectrogram with Mask', fontsize=35, pad=20)  # Added pad for title
        self._add_mask_overlay(axs[0], mask_np[0], spec_np[0], mask_bar_height)

        # Plot 2: Model Prediction with Mask
        axs[1].imshow(output[0].T, aspect='auto', origin='lower')
        axs[1].set_title('Model Prediction with Mask', fontsize=35, pad=20)  # Added pad for title
        self._add_mask_overlay(axs[1], mask_np[0], spec_np[0], mask_bar_height)

        # Save the figure
        plt.savefig(os.path.join(self.predictions_subfolder_path, f'MSE_Visualization_{step}.png'), format="png")
        plt.close(fig)

    def _add_mask_overlay(self, axis, mask, data, mask_bar_height):
        # Get the current y-axis limit to position the mask bar within the plot area
        y_min, y_max = axis.get_ylim()
        mask_bar_position = y_max - mask_bar_height  # Position at the top inside the plot

        mask_colormap = ['red' if m == 1 else 'none' for m in mask]
        for idx, color in enumerate(mask_colormap):
            if color == 'red':
                # Create a rectangle with the bottom left corner at (idx, mask_bar_position)
                axis.add_patch(plt.Rectangle((idx, mask_bar_position), 1, mask_bar_height, 
                                            edgecolor='none', facecolor=color))

    def visualize_masked_predictions(self, step, spec, output, mask, all_outputs):
        self.model.eval()
        with torch.no_grad():
            self.visualize_mse(output=output, mask=mask, spec=spec, step=step)
            # Create a large canvas of intermediate outputs
            self.create_large_canvas(all_outputs)
            # Save the large canvas
            # plt.savefig(os.path.join(self.predictions_subfolder_path, f'Intermediate Outputs_{step}.eps'), format="eps", dpi=300)
            plt.savefig(os.path.join(self.predictions_subfolder_path, f'Intermediate Outputs_{step}.png'), format="png")
            plt.close()
                
    def validate_model(self, step, spec):
        self.model.eval()
        with torch.no_grad():
            # Fetch the next batch from the validation set
            spec = spec.to(self.device)

            # Forward pass
            output, mask, _, intermediate_outputs = self.model.train_forward(spec)

            if step % self.eval_interval == 0 or step == 0:
                self.visualize_masked_predictions(step, spec, mask=mask, output=output, all_outputs=intermediate_outputs)


            # Calculate loss and accuracy
            val_loss, masked_seq_acc, unmasked_seq_acc, *rest = self.model.mse_loss(predictions=output, spec=spec , mask=mask, intermediate_layers=intermediate_outputs)

            # Convert to scalar values
            avg_val_loss = val_loss.item()
            avg_masked_seq_acc = masked_seq_acc.item()
            avg_unmasked_seq_acc = unmasked_seq_acc.item()

        return avg_val_loss, avg_masked_seq_acc, avg_unmasked_seq_acc

    def moving_average(self, values, window):
        """Simple moving average over a list of values"""
        if len(values) < window:
            # Return an empty list or some default value if there are not enough values to compute the moving average
            return []
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, 'valid')
        return sma.tolist()

    def train(self, continue_training=False, training_stats=None, last_step=0):
        step = last_step + 1 if continue_training else 0

        scaler = GradScaler()

        if continue_training:
            # Ensure lists are being used for these statistics
            raw_loss_list = training_stats.get('training_loss', [])
            raw_val_loss_list = training_stats.get('validation_loss', [])
            raw_masked_seq_acc_list = training_stats.get('masked_seq_acc', [])
            raw_unmasked_seq_acc_list = training_stats.get('unmasked_seq_acc', [])
            steps_since_improvement = training_stats.get('steps_since_improvement', 0)
            best_val_loss = training_stats.get('best_val_loss', float('inf'))
        else:
            # Initialize all metrics as lists
            raw_loss_list = []
            raw_val_loss_list = []
            raw_masked_seq_acc_list = []
            raw_unmasked_seq_acc_list = []
            steps_since_improvement = 0
            best_val_loss = float('inf')

        train_iter = iter(self.train_iter)
        test_iter =  iter(self.test_iter)

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        while step < self.max_steps:
            try:
                spec, ground_truth = next(train_iter)
                validation_spec, validation_ground_truth = next(test_iter)
        
            except Exception as e:
                # This block will execute if there is any exception in the try block
                print(f"An error occurred: {e}")
                continue

            spec = spec.to(self.device)
            ground_truth = ground_truth.to(self.device)

            self.model.train()

            with autocast():
                output, mask, _, intermediate_outputs = self.model.train_forward(spec)
                loss, *rest = self.model.mse_loss(predictions=output, spec=spec, mask=mask, intermediate_layers=intermediate_outputs)

            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.scheduler.step()

            # Accumulate raw losses
            raw_loss_list.append(loss.item())

            # Perform validation and accumulate metrics
            with autocast():
                val_loss, avg_masked_seq_acc, avg_unmasked_seq_acc = self.validate_model(step, spec=validation_spec)
                
            raw_val_loss_list.append(val_loss)
            raw_masked_seq_acc_list.append(avg_masked_seq_acc)
            raw_unmasked_seq_acc_list.append(avg_unmasked_seq_acc)

            if step % self.eval_interval == 0 or step == 0:
                # Apply moving average if there are enough data points
                if len(raw_loss_list) >= self.trailing_avg_window:
                    smoothed_training_loss = self.moving_average(raw_loss_list, self.trailing_avg_window)[-1]
                    smoothed_val_loss = self.moving_average(raw_val_loss_list, self.trailing_avg_window)[-1]
                    smoothed_masked_seq_acc = self.moving_average(raw_masked_seq_acc_list, self.trailing_avg_window)[-1]
                    smoothed_unmasked_seq_acc = self.moving_average(raw_unmasked_seq_acc_list, self.trailing_avg_window)[-1]
                else:
                    # Use the current values if not enough data points for moving average
                    smoothed_training_loss = raw_loss_list[-1]
                    smoothed_val_loss = val_loss
                    smoothed_masked_seq_acc = avg_masked_seq_acc
                    smoothed_unmasked_seq_acc = avg_unmasked_seq_acc

                print(f'Step [{step}/{self.max_steps}], '
                    f'Smoothed Training Loss: {smoothed_training_loss:.4f}, '
                    f'Smoothed Validation Loss: {smoothed_val_loss:.4f}, '
                    f'Smoothed Masked Seq Acc: {smoothed_masked_seq_acc:.4f}, '
                    f'Smoothed Unmasked Seq Acc: {smoothed_unmasked_seq_acc:.4f}')

                # Update best_val_loss and steps_since_improvement
                if smoothed_val_loss < best_val_loss:
                    best_val_loss = smoothed_val_loss
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1

                if self.early_stopping and steps_since_improvement >= self.patience:
                    print(f"Early stopping triggered at step {step}. No improvement for {self.patience} evaluation intervals.")
                    break

            # Save model and statistics
            if step % self.save_interval == 0 or step == self.max_steps - 1:
                current_training_stats = {
                    'step': step,
                    'training_loss': raw_loss_list,
                    'masked_seq_acc': raw_masked_seq_acc_list,
                    'unmasked_seq_acc': raw_unmasked_seq_acc_list,
                    'validation_loss': raw_val_loss_list,
                    'steps_since_improvement': steps_since_improvement,
                    'best_val_loss': best_val_loss
                }
                self.save_model(step, current_training_stats)

            step += 1
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        # print("")
        # print(f"number of reinit of dataloader {self.count_reinit}")

    def plot_results(self, save_plot=True, config=None):
        # Load the consolidated training statistics
        stats_filename = "training_statistics.json"
        stats_filepath = os.path.join(self.experiment_dir, stats_filename)
        
        with open(stats_filepath, 'r') as json_file:
            training_stats = json.load(json_file)
        
        steps = list(range(len(training_stats['training_loss'])))
        training_losses = training_stats['training_loss']
        validation_losses = training_stats['validation_loss']

        masked_seq_acc = training_stats['masked_seq_acc']
        unmasked_seq_acc = training_stats['unmasked_seq_acc']
        # Apply moving average to smooth the metrics
        smoothed_training_losses = self.moving_average(training_losses, self.trailing_avg_window)
        smoothed_validation_losses = self.moving_average(validation_losses, self.trailing_avg_window)
        smoothed_masked_seq_acc = self.moving_average(masked_seq_acc, self.trailing_avg_window)
        smoothed_unmasked_seq_acc = self.moving_average(unmasked_seq_acc, self.trailing_avg_window)

        # Adjust the steps for smoothed data
        smoothed_steps = steps[self.trailing_avg_window - 1:]

        plt.figure(figsize=(16, 12))

        # Plot 1: Smoothed Training and Validation Loss
        plt.subplot(2, 2, 1)
        plt.plot(smoothed_steps, smoothed_training_losses, label='Smoothed Training Loss')
        plt.plot(smoothed_steps, smoothed_validation_losses, label='Smoothed Validation Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Smoothed Training and Validation Loss')
        plt.legend()

        # Plot 2: Smoothed Masked and Unmasked Seq Acc
        plt.subplot(2, 2, 2)
        plt.plot(smoothed_steps, smoothed_masked_seq_acc, label='Smoothed Masked Seq Loss')
        plt.plot(smoothed_steps, smoothed_unmasked_seq_acc, label='Smoothed Unmasked Seq Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Masked and Unmasked Validation Loss')
        plt.legend()

        # Plot 3: Raw Training and Validation Loss
        plt.subplot(2, 2, 3)
        plt.plot(steps, training_losses, label='Raw Training Loss', alpha=0.7)
        plt.plot(steps, validation_losses, label='Raw Validation Loss', alpha=0.7)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Raw Training and Validation Loss')
        plt.legend()

        plt.tight_layout()

        if save_plot:
            plt.savefig(os.path.join(self.experiment_dir, 'training_validation_loss_plots.png'))
        else:
            plt.show()