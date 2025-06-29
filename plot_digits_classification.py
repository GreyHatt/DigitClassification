"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Standard scientific Python imports

# Import datasets, classifiers and performance metrics
from sklearn import datasets
from utils import (
    plot_digits,
    classifiction_and_plot_predicted,
    print_classication_report,
    display_confusion_matrix,
    reprint_classification_report,
)

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.
digits = datasets.load_digits()
plot_digits(digits)

y_test, predicted, clf = classifiction_and_plot_predicted(digits)

print_classication_report(clf, y_test, predicted)

disp = display_confusion_matrix(y_test, predicted)

reprint_classification_report(disp)
