import numpy as np
import matplotlib.pyplot as plt


def __print_bitmap(bitmap: iter, label, prediction, width: int = 28, height: int = 28):
    """
    Plots bitmap.
    :param bitmap: iter
        1d-array that represent the bitmap.
    :param width: int
        Width of the bitmap.
    :param height: int
        Height of the bitmap.
    """
    plt.title("Label: %d, Prediction: %d" % (label, prediction))

    rows = np.empty(shape=(0, width))

    for h in range(height):
        rows = np.append(rows, [bitmap[h * width: (h + 1) * width]], axis=0)

    plt.imshow(rows)
    plt.gray()
    plt.show()


def print_wrong_predictions(src_data: iter, predictions: iter, labels: iter, top: int, bitmap: bool = False):
    """
    Print/plot wrongly predicted results.
    :param src_data: iter
        2d-array with source data (if bitmap=True it is bitmaps represented by 1d-arrays).
    :param predictions: iter
        Predicted labels.
    :param labels: iter
        Correct labels.
    :param top: int (default=-1)
        How many wrong labels to print (-1 prints all)
    :param bitmap: bool (default=False)
        True if src_data is bitmap
    """
    j = 0

    for i, label in enumerate(labels):
        if j >= top >= 0:
            break

        if predictions[i] != label:
            j += 1

            if bitmap:
                plt.figure().canvas.set_window_title('%d/%d' % (j, top))
                __print_bitmap(src_data[i], labels[i], predictions[i])
            else:
                print('Label: %s Predicted: %s Msg: %s' % (label, predictions[i], src_data[i]))
