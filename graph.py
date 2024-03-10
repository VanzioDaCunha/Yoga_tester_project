import matplotlib.pyplot as plt
import numpy as np
import itertools

# plt.style.use('dark_background')


def plot_history(history):
    acc = history.history["accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    val_accuracy = history.history["val_accuracy"]

    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, "b", label="traning_acc")
    plt.plot(x, val_accuracy, "r", label="traning_acc")
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(x, loss, "b", label="traning_acc")
    plt.plot(x, val_loss, "r", label="traning_acc")
    plt.title("Loss")
    plt.show()


def plot_confusion_matrix(cm, class_names, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
        This function plots the confusion matrix with various customization options.

        Args:
            cm (numpy.array): The confusion matrix data.
            class_names (list): List of class names (strings) for labels.
            normalize (bool, optional): If True, normalize the data into proportions. Defaults to False.
            title (str, optional): Title for the plot. Defaults to 'Confusion matrix'.
            cmap (matplotlib.colors.Colormap, optional): Colormap for the heatmap. Defaults to plt.cm.Blues.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'  # Adjust format string for displaying decimals in percentages
    else:
        fmt = 'd'  # Use 'd' for displaying raw integer counts

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.grid(False)
    plt.tight_layout()
    plt.show()
