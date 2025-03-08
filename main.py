import numpy as np

class CNN():
    def __init__(self, input_shape, learning_rate=0.01):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.layers = []
        self.loss_function = None
        self.optimizer = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X): # one forward pass through the neural network
        pass

    def backward(self, y_true): # one backward pass through the neural network
        pass

    def train(self, X, y, epochs): # training the dataset includes 
        pass

    def predict(self, X):
<<<<<<< Updated upstream
        pass
=======
        self.eval()  # Switch to evaluation mode
        with torch.no_grad():
            outputs = torch.sigmoid(self.forward(X)).squeeze(1)
            return (outputs > 0.4).int()  # Convert to binary (0 or 1)

if __name__ == '__main__':
    print('Starting training')
    model = CNN()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    
    
    preprocessor = Preprocessor()
    dataloaders = preprocessor.process()
    best_val_loss = float('inf')

    optimizer=torch.optim.AdamW(model.parameters())
    bce = nn.BCEWithLogitsLoss()

    model.to(device)
    epochs = 3

    # ----------------TRAIN STARTS HERE----------------
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        total_loss = 0.0
        for inputs, labels in dataloaders['train']:
        # Move data to GPU if available
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = bce(outputs, labels.float().unsqueeze(1))  # Fix: add .unsqueeze(1)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
        epoch_loss = total_loss / len(dataloaders['train'].dataset)
        print(f"Training Loss: {epoch_loss:.4f}")

         # ----------------VALIDATION STARTS HERE----------------
                # Validation loop
        model.eval()  # Switch to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        # Initialize confusion matrix (2x2 for binary classification)
        confusion_matrix = [[0, 0], [0, 0]]

        with torch.no_grad():
            for inputs, labels in dataloaders['validate']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = bce(outputs, labels.float().unsqueeze(1))
                val_loss += loss.item() * inputs.size(0)

                # Calculate accuracy and confusion matrix
                preds = (torch.sigmoid(outputs) > 0.5).int()
                correct += (preds.squeeze(1) == labels).sum().item()
                total += labels.size(0)

                # Update confusion matrix
                for pred, label in zip(preds.squeeze(1).cpu().numpy(), labels.cpu().numpy()):
                    if label == 0:
                        if pred == 0:
                            confusion_matrix[0][0] += 1  # True Negative
                        else:
                            confusion_matrix[0][1] += 1  # False Positive
                    else:
                        if pred == 0:
                            confusion_matrix[1][0] += 1  # False Negative
                        else:
                            confusion_matrix[1][1] += 1  # True Positive

        avg_val_loss = val_loss / len(dataloaders['validate'].dataset)
        val_accuracy = correct / total * 100
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(f"TN: {confusion_matrix[0][0]}  FP: {confusion_matrix[0][1]}")
        print(f"FN: {confusion_matrix[1][0]}  TP: {confusion_matrix[1][1]}\n")

        # Save model if validation loss improves
        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), 'best_skin_cancer_model_state.pth')
            best_val_loss = avg_val_loss
            print("Best model saved based on validation loss.")


    print("Training complete.")
    # ----------------TRAIN ENDS HERE----------------
>>>>>>> Stashed changes
