import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os 
import json 
import numpy as np 
from sklearn.decomposition import PCA
from torch.cuda.amp import autocast, GradScaler  # Import autocast and GradScaler for mixed precision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

class ModifiedCrossEntropyLoss(nn.Module):
    def __init__(self, similarity_penalty_weight=0.1, entropy_weight=0.01, temperature=1.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.similarity_penalty_weight = similarity_penalty_weight
        self.entropy_weight = entropy_weight
        self.temperature = temperature

    def forward(self, predictions, targets):
        # predictions shape: [batch_size, num_classes, sequence_length]
        # targets shape: [batch_size, sequence_length]
        batch_size, num_classes, seq_length = predictions.shape
        
        # Apply temperature scaling
        scaled_predictions = predictions / self.temperature
        
        # Reshape predictions to [batch_size * sequence_length, num_classes]
        predictions_reshaped = scaled_predictions.permute(0, 2, 1).reshape(-1, num_classes)
        
        # Reshape targets to [batch_size * sequence_length]
        targets_reshaped = targets.reshape(-1)

        # Calculate cross-entropy loss
        ce_loss = self.ce_loss(predictions_reshaped, targets_reshaped)

        # Calculate pairwise cosine similarity between consecutive prediction vectors
        predictions_norm = F.normalize(scaled_predictions, p=2, dim=1)
        cosine_sim = F.cosine_similarity(predictions_norm[:, :, :-1].unsqueeze(2), 
                                         predictions_norm[:, :, 1:].unsqueeze(1), 
                                         dim=3)
        
        # Average similarity across batch, classes, and sequence
        avg_similarity = cosine_sim.mean()

        # Penalty is inverse of similarity (1 - similarity) to encourage dissimilarity
        similarity_penalty = 1 - avg_similarity

        # Calculate entropy
        probs = F.softmax(scaled_predictions, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()

        # Combine losses
        total_loss = ce_loss + self.similarity_penalty_weight * similarity_penalty - self.entropy_weight * entropy

        return total_loss, ce_loss, similarity_penalty



    # def forward(self, predictions, targets):
    #     # predictions shape: [batch_size, num_classes, sequence_length]
    #     # targets shape: [batch_size, sequence_length]

    #     batch_size, num_classes, seq_length = predictions.shape
        
    #     # Reshape predictions to [batch_size * sequence_length, num_classes]
    #     predictions_reshaped = predictions.permute(0, 2, 1).reshape(-1, num_classes)
        
    #     # Reshape targets to [batch_size * sequence_length]
    #     targets_reshaped = targets.reshape(-1)

    #     # Calculate cross-entropy loss
    #     ce_loss = self.ce_loss(predictions_reshaped, targets_reshaped)

    #     # Calculate state switching penalty based on argmax logits
    #     argmax_predictions = torch.argmax(predictions, dim=1)

    #     # Calculate the number of switches using vectorized operations
    #     switches = (argmax_predictions[:, 1:] != argmax_predictions[:, :-1]).sum().item()
    
    #     switching_penalty = switches / (batch_size * seq_length)

    #     # Combine losses
    #     total_loss = ce_loss + self.switching_penalty_weight * switching_penalty

    #     total_loss = ce_loss 

    #     return total_loss, ce_loss, switching_penalty


class LinearProbeModel(nn.Module):
    def __init__(self, num_classes, model_type="neural_net", model=None, freeze_layers=True, layer_num=-1, layer_id="feed_forward_output_relu", classifier_dims=2):
        super(LinearProbeModel, self).__init__()
        self.model_type = model_type
        self.freeze_layers = freeze_layers
        self.layer_num = layer_num
        self.layer_id = layer_id
        self.model = model
        self.classifier_dims = classifier_dims
        self.num_classes = num_classes 

        # Define a 3-layer decoder with ReLU activations
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dims, 128),    
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )
        
        if freeze_layers and model_type == "neural_net":
            self.freeze_transformer_blocks(self.model, freeze_up_to_block=2)
            # self.freeze_all_but_classifier(self.model)
        if model_type == "pca":
            self.pca = PCA(n_components=classifier_dims, random_state=42)

    def forward(self, input):
        # with autocast():  # Use autocast for the forward pass to enable mixed precision
        if self.model_type == "neural_net":
            outputs, layers = self.model.inference_forward(input)

            if self.layer_id == "embedding":
                features = outputs 
            else:
                features = layers[self.layer_num][self.layer_id]
            logits = self.classifier(features)

        elif self.model_type == "umap":
            # reformat for UMAP 
            # remove channel dim intended for conv network 
            input = input[:,0,:,:]
            output_shape = input.shape
            input = input.reshape(-1,input.shape[1])
            input = input.detach().cpu().numpy()
            reduced = self.model.transform(input)
            reduced = torch.Tensor(reduced).to(self.device)
            logits = self.classifier(reduced)
            
            # shape is batch x num_classes (how many classes in the dataset) x sequence length 
            logits = logits.reshape(output_shape[0],output_shape[2],self.num_classes)

        elif self.model_type == "pca":
            # reformat for UMAP 
            # remove channel dim intended for conv network 
            input = input[:,0,:,:]
            output_shape = input.shape
            input = input.reshape(-1,input.shape[1])
            input = input.detach().cpu().numpy()
            reduced = self.pca.fit_transform(input)
            reduced = torch.Tensor(reduced).to(self.device)
            logits = self.classifier(reduced)
            # shape is batch x num_classes (how many classes in the dataset) x sequence length
            logits = logits.reshape(output_shape[0],output_shape[2],self.num_classes)

        elif self.model_type == "raw":
            # reformat for UMAP 
            # remove channel dim intended for conv network 
            input = input[:,0,:,:]
            output_shape = input.shape
            input = input.reshape(-1,input.shape[1])
            logits = self.classifier(input)
            # shape is batch x num_classes (how many classes in the dataset) x sequence length
            logits = logits.reshape(output_shape[0],output_shape[2],self.num_classes)

        return logits
    
    def cross_entropy_loss(self, predictions, targets):
        
        loss_fn = ModifiedCrossEntropyLoss()
        total_loss, ce_loss, switching_penalty = loss_fn(predictions, targets)

        return total_loss

    def freeze_all_but_classifier(self, model):
        for name, module in model.named_modules():
            if name != "classifier":
                for param in module.parameters():
                    param.requires_grad = False

        for param in self.classifier.parameters():
            param.requires_grad = True

    def freeze_transformer_blocks(self, model, freeze_up_to_block):
        """
        Freeze all layers up to a certain transformer block, including conv and projection layers.
        """
        freeze = True
        for name, module in model.named_modules():
            if "transformer_encoder" in name and '.' in name:
                block_num = int(name.split('.')[1])
                if block_num >= freeze_up_to_block:
                    freeze = False

            if freeze:
                for param in module.parameters():
                    param.requires_grad = False

        # Ensure the classifier's parameters are always trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

    # overwrite way we have access to the models device state 
    def to(self, device):
        self.device = device
        return super(LinearProbeModel, self).to(device)

class LinearProbeTrainer():
    def __init__(self, model, train_loader, test_loader, device, lr=1e-2, plotting=False, batches_per_eval=200, desired_total_batches=1e4, patience=8, use_tqdm=True, moving_avg_window = 1):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-2, weight_decay=0.0)
        self.plotting = plotting
        self.batches_per_eval = batches_per_eval
        self.desired_total_batches = desired_total_batches
        self.patience = patience
        self.use_tqdm = use_tqdm
        self.moving_avg_window = moving_avg_window  # Window size for moving average
        self.scaler = GradScaler()  # Initialize GradScaler for mixed precision
        self.weight_grad_monitor = WeightGradientMonitor(self.model)
        self.save_dir = "weight_gradient_plots"
        os.makedirs(self.save_dir, exist_ok=True)

    def frame_error_rate(self, y_pred, y_true):
        y_pred = y_pred.permute(0,2,1)
        y_pred = y_pred.argmax(-1)

        mismatches = (y_pred != y_true).float()
        error = mismatches.sum() / y_true.numel()
        return error * 100

    def validate_model(self, test_iter):
        self.model.eval()
        total_val_loss = 0
        total_frame_error = 0

        with torch.no_grad():
            try:
                spectrogram, label, vocalization = next(test_iter)
            except StopIteration:
                test_iter = iter(self.test_loader)  # Reinitialize the iterator
                spectrogram, label, vocalization = next(test_iter)

            spectrogram, label, vocalization = spectrogram.to(self.device), label.to(self.device), vocalization.to(self.device)
            # with autocast():  # Use autocast for the validation pass
            output = self.model.forward(spectrogram)
            label = label.argmax(dim=-1)
            output = output.permute(0,2,1)

            loss = self.model.cross_entropy_loss(predictions=output, targets=label)
            total_val_loss = loss.item()
            total_frame_error = self.frame_error_rate(output, label).item()

        return total_val_loss, total_frame_error

    def moving_average(self, values, window):
        """Simple moving average over a list of values"""
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, 'valid')
        return sma.tolist()

    def train(self):
        total_batches = 0
        best_val_loss = float('inf')
        num_val_no_improve = 0
        stop_training = False

        raw_loss_list, raw_val_loss_list, raw_frame_error_rate_list = [], [], []
        moving_avg_val_loss_list, moving_avg_frame_error_list = [], []

        train_iter = iter(self.train_loader)
        test_iter = iter(self.test_loader)

        while total_batches < self.desired_total_batches and not stop_training:
            try:
                spectrogram, label, vocalization = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)  # Reinitialize the iterator
                spectrogram, label, vocalization = next(train_iter)

            spectrogram, label, vocalization = spectrogram.to(self.device), label.to(self.device), vocalization.to(self.device)
            self.optimizer.zero_grad()
         
            output = self.model.forward(spectrogram)

            label = label.argmax(dim=-1)
            output = output.permute(0,2,1)

            loss = self.model.cross_entropy_loss(predictions=output, targets=label)

            loss.backward()
            self.optimizer.step()

            total_batches += 1
            if total_batches % self.batches_per_eval == 0:
                avg_val_loss, avg_frame_error = self.validate_model(test_iter)

                raw_loss_list.append(loss.item())
                raw_val_loss_list.append(avg_val_loss)
                raw_frame_error_rate_list.append(avg_frame_error)

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
                            stop_training = True
                            break

                if self.use_tqdm: 
                    print(f'Step {total_batches}: Train Loss {loss.item():.4f} FER = {avg_frame_error:.2f}%, Val Loss = {avg_val_loss:.4f}')

            if stop_training:
                break
        self.weight_grad_monitor.plot(self.save_dir)


        if self.plotting:
            self.plot_results(raw_loss_list, moving_avg_val_loss_list, moving_avg_frame_error_list)

    def plot_results(self, loss_list, val_loss_list, frame_error_rate_list):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(loss_list, label='Training Loss')
        plt.plot(val_loss_list, label='Validation Loss')
        plt.title('Loss over Steps')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(frame_error_rate_list, label='Frame Error Rate', color='red')
        plt.title('Frame Error Rate over Steps')
        plt.xlabel('Steps')
        plt.ylabel('Error Rate (%)')

        plt.legend()

        plt.tight_layout()
        plt.show()


class ModelEvaluator:
    def __init__(self, model, test_loader, num_classes=21, device='cuda:0', use_tqdm=True, filter_unseen_classes=False, train_dir=None):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.device = device
        self.use_tqdm = use_tqdm
        self.filter_unseen_classes = filter_unseen_classes
        self.seen_classes = set(range(num_classes))  # Assume all classes are seen by default
        if filter_unseen_classes and train_dir:
            self.seen_classes = self.count_labels_in_training_set(train_dir)

    def count_labels_in_training_set(self, train_dir):
        seen_classes = set()
        for file_name in os.listdir(train_dir):
            if file_name.endswith('.npz'):
                data = np.load(os.path.join(train_dir, file_name))
                labels = data['labels']
                unique_labels = set(np.unique(labels))
                seen_classes.update(unique_labels)
        return seen_classes

    def validate_model_multiple_passes(self, num_passes=1, max_batches=1e4, spec_height=196, context=1000):
        self.model.eval()
        errors_per_class = [0] * self.num_classes
        correct_per_class = [0] * self.num_classes
        total_frames = 0
        total_errors = 0

        total_iterations = max(max_batches, len(self.test_loader))
        progress_bar = tqdm(total=total_iterations, desc="Evaluating", unit="batch") if self.use_tqdm else None
        for _ in range(num_passes):
            with torch.no_grad():
                for spec, label, vocalization in self.test_loader:
                    spec, label, vocalization = spec.to(self.device), label.to(self.device), vocalization.to(self.device)
                    # Removed print statements as per instructions

                    print(f"spec shape: {spec.shape}, label shape: {label.shape}, vocalization shape: {vocalization.shape}")

                    # this is all done to ensure that all of the eval dataset is seen by the model 
                    # First, pad spec to the nearest multiple of 500
                    pad_size = (context - spec.size(1) % context) % context
                    spec = F.pad(spec, (0, 0, 0, pad_size))
                    label = F.pad(label, (0, 0, 0, pad_size))

                    # Now, reshape this to be [n x length x freq bins]
                    spec = spec.reshape(-1, context, spec_height)
                    label = label.reshape(-1, context, self.num_classes)

                    spec = spec.unsqueeze(1)

                    output = self.model.forward(spec.permute(0, 1, 3, 2))

                    label = label.squeeze(1)

                    predicted_labels = output.argmax(dim=-1)
                    true_labels = label.argmax(dim=-1)

                    correct = (predicted_labels == true_labels)
                    incorrect = ~correct

                    for cls in range(self.num_classes):
                        if self.filter_unseen_classes and cls not in self.seen_classes:
                            continue  # Skip classes not seen in training

                        class_mask = true_labels == cls
                        incorrect_class = incorrect & class_mask

                        errors_per_class[cls] += incorrect_class.sum().item()
                        correct_per_class[cls] += (correct & class_mask).sum().item()

                        total_frames += class_mask.sum().item()
                        total_errors += incorrect_class.sum().item()

                    if progress_bar is not None:
                        progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

        class_frame_error_rates = {
            cls: (errors / (errors + correct) * 100 if errors + correct > 0 else float('nan'))
            for cls, (errors, correct) in enumerate(zip(errors_per_class, correct_per_class))
        }
        total_frame_error_rate = (total_errors / total_frames * 100 if total_frames > 0 else float('nan'))
        return class_frame_error_rates, total_frame_error_rate

    def save_results(self, class_frame_error_rates, total_frame_error_rate, folder_path, layer_id=None, layer_num=None):
        # Conditional filename based on whether layer_id and layer_num are provided
        if layer_id is not None and layer_num is not None:
            suffix = f'_{layer_id}_{layer_num}'
        else:
            suffix = ''

        # Save plot
        plot_filename = f'frame_error_rate_plot{suffix}.png'
        self.plot_error_rates(class_frame_error_rates, plot_filename, folder_path)

        # Save data to JSON
        results_data = {
            'class_frame_error_rates': class_frame_error_rates,
            'total_frame_error_rate': total_frame_error_rate
        }
        json_filename = f'results{suffix}.json'
        with open(os.path.join(folder_path, json_filename), 'w') as file:
            json.dump(results_data, file)

    def plot_error_rates(self, class_frame_error_rates, plot_filename, save_path):
        if self.filter_unseen_classes:
            # Filter out unseen classes before plotting
            filtered_class_frame_error_rates = {cls: rate for cls, rate in class_frame_error_rates.items() if cls in self.seen_classes}
        else:
            filtered_class_frame_error_rates = class_frame_error_rates

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(filtered_class_frame_error_rates)), filtered_class_frame_error_rates.values(), color='skyblue')
        plt.xlabel('Class', fontsize=15)
        plt.ylabel('Frame Error Rate (%)', fontsize=15)
        plt.title(f'Frame Error Rates - {plot_filename.replace(".png", "")}', fontsize=15)
        plt.xticks(range(len(filtered_class_frame_error_rates)), filtered_class_frame_error_rates.keys(), fontsize=12, rotation=45)
        plt.yticks(fontsize=12)
        plt.ylim(0, max(filtered_class_frame_error_rates.values()) + 5)  # Ensure the y-axis goes a bit beyond the max value for better visualization
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, plot_filename))
        plt.close()

    def plot_spectrogram_with_labels(self, spec, true_labels, predicted_labels, save_path, filename):
        plt.figure(figsize=(12, 6))
        
        # Plot the spectrogram
        plt.subplot(3, 1, 1)
        plt.imshow(spec.squeeze().cpu().numpy(), aspect='auto', origin='lower')
        plt.title('Spectrogram')
        plt.colorbar(format='%+2.0f dB')

        # Plot the ground truth labels
        plt.subplot(3, 1, 2)
        plt.imshow(true_labels.unsqueeze(0).cpu().numpy(), aspect='auto', cmap='tab20', origin='lower')
        plt.title('Ground Truth Labels')
        plt.colorbar(ticks=range(self.num_classes))

        # Plot the predicted labels
        plt.subplot(3, 1, 3)
        plt.imshow(predicted_labels.unsqueeze(0).cpu().numpy(), aspect='auto', cmap='tab20', origin='lower')
        plt.title('Predicted Labels')
        plt.colorbar(ticks=range(self.num_classes))

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, filename))
        plt.close()