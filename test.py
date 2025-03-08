import torch
from torch import nn
from main import CNN
from preprocessor import Preprocessor


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
# Initialize a new model instance first
    model = CNN()
    model.load_state_dict(torch.load('best_skin_cancer_model_state.pth'))
    # model.load_state_dict(torch.load('current_model.pth'))

    model.to(device)  # Move to the correct device (CPU or GPU)
    model.eval()       # Set to evaluation mode
        # Set to evaluation mode
    
    preprocessor = Preprocessor()
    dataloaders = preprocessor.process()

    epochs = 5
    # Load model

    total_predictions = 0
    correct_predictions = 0

    # Initialize confusion matrix (2x2 for binary classification)
    confusion_matrix = [[0, 0], [0, 0]]

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)  # Move images to the same device as the model
            labels = labels.to(device)  # Move labels too (for comparison)
            
            # Use the predict method
            predictions = model.predict(inputs)

            # Update confusion matrix
            for pred, label in zip(predictions.cpu().numpy(), labels.cpu().numpy()):
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

            
            # Print results
            # print("Predictions:", predictions)
            # print("Actual labels:", labels)
            correct = (predictions == labels).sum().item()  # Fix: no .squeeze(1) needed
            correct_predictions += correct
            total = labels.size(0)
            total_predictions += total
            accuracy = correct / total * 100
            print(f"Batch[{i}] Accuracy: {accuracy:.2f}%")
            # Stop after first batch for demo
    accuracy = correct_predictions / total_predictions * 100
    print(f"Total Accuracy: {accuracy:.2f}%")

    TN = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TP = confusion_matrix[1][1]

    print("\nConfusion Matrix:")
    print(f"TN: {TN}  FP: {FP}")
    print(f"FN: {FN}  TP: {TP}\n")

    print(f"Recall: {TP/(TP+FN)}")
    print(f"Precision: {TP/(TP+FP)}")