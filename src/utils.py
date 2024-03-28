import matplotlib.pyplot as plt
import numpy as np

def evaluate_save_model(history, output_dir):
    if isinstance(history, dict):
        model_train_history = history
    elif hasattr(history, 'history'):
        model_train_history = history.history
    else:
        raise ValueError("Invalid history object provided.")

    num_epochs = len(model_train_history.get("loss", []))

    # Plot training results
    fig = plt.figure(figsize=(15, 5))
    axs = fig.add_subplot(1, 2, 1)
    axs.set_title('Loss')
    for metric in ["loss", "val_loss"]:
        axs.plot(np.arange(0, num_epochs), model_train_history.get(metric, []), label=metric)
    axs.legend()

    axs = fig.add_subplot(1, 2, 2)
    axs.set_title('Accuracy')
    for metric in ["accuracy", "val_accuracy"]:
        axs.plot(np.arange(0, num_epochs), model_train_history.get(metric, []), label=metric)
    axs.legend()

    plt.savefig(output_dir + '/training_results.png')
    plt.show()
