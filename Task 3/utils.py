import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from itertools import cycle



def evaluate(y_true, y_pred, model_name):
    # Calculate and print classification report
    report = classification_report(y_true, y_pred)
    print(f"{model_name} Classification Report:\n")
    print(report)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr', 
                xticklabels=sorted(set(y_true)), 
                yticklabels=sorted(set(y_true)))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    
    
    
def ROC_AUC(y_test, y_proba, data):
    
    fpr_lr = {}
    tpr_lr = {}
    roc_auc_lr = {}
    plt.figure(figsize=(12, 6))

    for i in range(len(data.target_names)):
        fpr_lr[i], tpr_lr[i], _ = roc_curve((y_test == i).astype(int), y_proba[i])
        roc_auc_lr[i] = auc(fpr_lr[i], tpr_lr[i])

    plt.subplot(1, 2, 2)
    for i in range(len(data.target_names)):
        plt.plot(fpr_lr[i], tpr_lr[i], label=f'Class {i} (AUC = {roc_auc_lr[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve for Custom Logistic Regression')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
    
    
def plot_roc_curves(y_test, y_proba, model_name):
   
    y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_binarized.shape[1]
        
        # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot all ROC curves in one plot
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.show()
    
    
    




    