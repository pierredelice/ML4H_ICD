
import matplotlib.pyplot as plt
import time

def plot_training_and_evaluation_metrics(training_losses, training_accuracy, evaluation_accuracy, evaluation_precision, evaluation_recall, evaluation_f1):
    # Create a figure for the plots
    plt.figure(figsize=(15, 8))

    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(training_losses) + 1), training_losses, marker='o', label='Training Loss', color='b')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/training_loss.png')  # Save the training loss plot

    # Plot training accuracy
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(training_accuracy) + 1), training_accuracy, marker='o', label='Training Accuracy', color='g')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/training_accuracy.png')  # Save the training accuracy plot

    # Plot evaluation accuracy
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(evaluation_accuracy) + 1), evaluation_accuracy, marker='o', label='Evaluation Accuracy', color='r')
    plt.title('Evaluation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/evaluation_accuracy.png')  # Save the evaluation accuracy plot

    # Plot evaluation precision, recall, and F1-score
    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(evaluation_precision) + 1), evaluation_precision, marker='o', label='Precision', color='c')
    plt.plot(range(1, len(evaluation_recall) + 1), evaluation_recall, marker='o', label='Recall', color='m')
    plt.plot(range(1, len(evaluation_f1) + 1), evaluation_f1, marker='o', label='F1-Score', color='y')
    plt.title('Evaluation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/evaluation_metrics.png')  # Save the evaluation metrics plot

    # Adjust layout and show all plots
    plt.tight_layout()
    time.sleep(2)
    plt.close()
