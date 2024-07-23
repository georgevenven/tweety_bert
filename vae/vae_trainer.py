import os
import numpy as np 
import torch
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import json

class ModelTrainer:
    def __init__(self, model, train_loader, test_loader, optimizer, device, 
             max_steps=10000, eval_interval=500, save_interval=1000, 
             weights_save_dir='saved_weights', 
             save_weights=True, overfit_on_batch=False, experiment_dir=None, loss_function=None, early_stopping=True, patience=8, trailing_avg_window=1000, path_to_prototype_clusters=None):

        self.overfit_on_batch = overfit_on_batch
        self.fixed_batch = None  # Will hold the batch data when overfitting
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.scheduler = StepLR(optimizer, step_size=10000, gamma=1)
        self.early_stopping = early_stopping
        self.patience = patience 
        self.trailing_avg_window = trailing_avg_window  # Window size for trailing average calculation

        self.loss_list = []
        self.val_loss_list = []
        self.sum_squared_weights_list = []
        self.masked_sequence_accuracy_list = []
        self.unmasked_sequence_accuracy_list = []
        self.val_masked_sequence_accuracy_list = []
        self.val_unmasked_sequence_accuracy_list = []
        self.avg_seq_acc = [] 

        self.save_interval = save_interval
        self.weights_save_dir = weights_save_dir
        self.save_weights = save_weights
        self.experiment_dir = experiment_dir  # Assuming the experiment dir is the parent of visualizations_save_dir
        
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
 
    def save_model(self, step):
        if self.save_weights:
            filename = f"model_step_{step}.pth"
            filepath = os.path.join(self.weights_save_dir, filename)
            torch.save(self.model.state_dict(), filepath)

    def save_images(self, x, x_recon, step):
        # Assuming x and x_recon are PyTorch tensors with shape [batch_size, channels, height, width]
        # Select the first image in the batch and remove the channel dimension if it's 1
        x = x[0].squeeze().cpu().detach().numpy()  # Convert to numpy array for plotting
        x_recon = x_recon[0].squeeze().cpu().detach().numpy()  # Convert to numpy array for plotting

        # Plotting
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Original image
        axs[0].imshow(x, aspect='auto', origin='lower')
        axs[0].set_title('Ground Truth')
        axs[0].axis('off')  # Hide axes for better visualization

        # Reconstructed image
        axs[1].imshow(x_recon, aspect='auto', origin='lower')
        axs[1].set_title('Reconstruction')
        axs[1].axis('off')  # Hide axes for better visualization

        # Save the figure
        plt.savefig(os.path.join(self.predictions_subfolder_path, f'MSE_Visualization_{step}.png'), format="png", dpi=300)
        plt.close(fig)
                
    def validate_model(self, step, test_iter):
        self.model.eval()
        with torch.no_grad():
            try:
                spec, ground_truth = next(test_iter)
            except StopIteration:
                test_iter = iter(self.test_loader)
                spec, ground_truth = next(test_iter)

            # Fetch the next batch from the validation set
            spec = spec.to(self.device)
            ground_truth = ground_truth.to(self.device)

            x_recon, latent_mu, latent_logvar = self.model.forward(spec)
            # There can be a variable number of variables returned
            loss = self.model.loss(recon_x=x_recon, mu=latent_mu, logvar=latent_logvar, x=spec)

            if step % self.eval_interval == 0 or step == 0:
                self.save_images(step=step, x=spec, x_recon=x_recon)

            # Convert to scalar values
            avg_val_loss = loss.item()

        return avg_val_loss

    def moving_average(self, values, window):
        """Simple moving average over a list of values"""
        if len(values) < window:
            # Return an empty list or some default value if there are not enough values to compute the moving average
            return []
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, 'valid')
        return sma.tolist()

    def train(self):
        step = 0
        best_val_loss = float('inf')  # Best validation loss seen so far
        steps_since_improvement = 0  # Counter for steps since last improvement

        train_iter = iter(self.train_loader)
        test_iter = iter(self.test_loader)

        # Initialize lists for storing metrics
        raw_loss_list = []
        raw_val_loss_list = []
        smoothed_val_loss_list = []

        while step < self.max_steps:
            try:
                spec, ground_truth = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                spec, ground_truth = next(train_iter)

            spec = spec.to(self.device)
            ground_truth = ground_truth.to(self.device)


            self.model.train()  # Explicitly set the model to training mode

            x_recon, latent_mu, latent_logvar = self.model.forward(spec)
            # There can be a variable number of variables returned
            loss = self.model.loss(recon_x=x_recon, mu=latent_mu, logvar=latent_logvar, x=spec)

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Store metrics after each step
            raw_loss_list.append(loss.item())

            # Your existing code where validation loss is computed
            val_loss  = self.validate_model(step, test_iter)
            raw_val_loss_list.append(val_loss)
            
            if step % self.save_interval == 0:
                self.save_model(step)

            if step >= self.max_steps:
                self.save_model(step)

            if step % self.eval_interval == 0 or step == 0:
                # Ensure val_loss_list has enough values before attempting to smooth
                if len(raw_val_loss_list) >= self.eval_interval:
                    smooth_loss = self.moving_average(raw_loss_list, self.eval_interval)
                    
                    smooth_val_loss = self.moving_average(self.val_loss_list, self.eval_interval)
                    
                    if len(raw_val_loss_list) >= self.eval_interval:
                        smooth_val_loss = self.moving_average(raw_val_loss_list, self.eval_interval)
                        smoothed_val_loss_list.append(smooth_val_loss[-1])
                        print(f'Step [{step}/{self.max_steps}]',
                            f'Training Loss: {smooth_loss[-1]:.4e}', 
                            f'Validation Loss: {smooth_val_loss[-1]:.4e}')
                    else:
                        print(f'Step [{step}/{self.max_steps}], '
                            f'Validation Loss: {raw_val_loss_list[-1]:.4e}')

                    if len(smoothed_val_loss_list) > 0:
                        current_smoothed_val_loss = smoothed_val_loss_list[-1]
                        is_best = current_smoothed_val_loss < best_val_loss

                        if is_best:
                            best_val_loss = current_smoothed_val_loss
                            steps_since_improvement = 0
                        else:
                            steps_since_improvement += 1

                        if self.early_stopping and steps_since_improvement >= self.patience:
                            print(f"Early stopping triggered at step {step}. No improvement for {self.patience} evaluation intervals.")
                            self.save_model(step)
                            break  # Exit the training loop

            step += 1


    def plot_results(self, save_plot=False, config=None, smoothing_window=100):
        # Calculate smoothed curves for the metrics
        smoothed_training_loss = self.moving_average(self.loss_list, smoothing_window)
        smoothed_validation_loss = self.moving_average(self.val_loss_list, smoothing_window)

        plt.figure(figsize=(16, 6))  # Adjusted the figure size

        # Plot 1: Training and Validation Loss
        plt.subplot(1, 2, 1)  # Adjusted for 2 plots instead of 3
        plt.plot(smoothed_training_loss, label='Smoothed Training Loss')
        plt.plot(smoothed_validation_loss, label='Smoothed Validation Loss')
        plt.legend()
        plt.title('Smoothed Training and Validation Loss')

        # Plot 2: Sum of Squared Weights per Step
        plt.subplot(1, 2, 2)  # Adjusted for 2 plots instead of 3
        plt.plot(self.sum_squared_weights_list, color='red', label='Sum of Squared Weights')
        plt.legend()
        plt.title('Sum of Squared Weights per Step')

        plt.tight_layout()
        
        if save_plot:
            if not os.path.exists(self.experiment_dir):
                os.makedirs(self.experiment_dir)
            plt.savefig(os.path.join(self.experiment_dir, 'smoothed_loss_accuracy_curves.png'))
        else:
            plt.show()

        # Prepare training statistics dictionary
        training_stats = {
            'smoothed_training_loss': smoothed_training_loss[-1] if smoothed_training_loss else None,
            'smoothed_validation_loss': smoothed_validation_loss[-1] if smoothed_validation_loss else None,
            'sum_squared_weights': self.sum_squared_weights_list[-1] if self.sum_squared_weights_list else None
        }

        # Save the training statistics as JSON
        stats_file_path = os.path.join(self.experiment_dir, 'training_statistics.json')
        with open(stats_file_path, 'w') as json_file:
            json.dump(training_stats, json_file, indent=4)

        print(f"Training statistics saved to {stats_file_path}")