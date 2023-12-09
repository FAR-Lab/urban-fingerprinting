# FARLAB - UrbanECG 
# Developer: @mattwfranchi
# Last Edited: 12/8/2023 

# This script houses a class to generate an ROC curve and AUC score for a given classification model trained with MMPretrain. 

# Import Packages
import os
import sys 

sys.path.append(os.path.join("..", "..", ".."))

from src.utils.logger import setup_logger 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


class AUC_ROC:

    def __init__(self, predictions_path):
            self.logger = setup_logger("AUC_ROC")
            self.logger.setLevel("INFO")
            self.logger.info("Initializing AUC_ROC class")

            # predictions should be a pkl file generated via mmpretrain/tools/test.py
            self.predictions = pd.read_pickle(predictions_path)

            # convert predictiosn to dataframe
            self.predictions = pd.DataFrame(self.predictions)

            # extract tensors in pred_label, pred_score, and gt_label columns
            self.predictions['pred_label'] = self.predictions['pred_label'].apply(lambda x: x.cpu().numpy()[0])
            self.predictions['pred_score'] = self.predictions['pred_score'].apply(lambda x: x.cpu().numpy()[0])
            self.predictions['gt_label'] = self.predictions['gt_label'].apply(lambda x: x.cpu().numpy()[0])

            # pred_score is closer to 0 for positive class, so we need to invert it
            self.predictions['pred_score'] = self.predictions['pred_score'].apply(lambda x: 1-x)

            self.logger.info("Predictions DataFrame Head:"+"\n"+self.predictions[self.predictions.gt_label==0].head().to_string()+"\n"+self.predictions[self.predictions.gt_label==1].head().to_string())

    def inspect_predictions(self):
         # distribution of pred_label 
        self.logger.info("Distribution of pred_label:" + "\n" + str(self.predictions['pred_label'].value_counts()))

        # distribution of gt_label 
        self.logger.info("Distribution of gt_label:" + "\n" + str(self.predictions['gt_label'].value_counts()))

        # true positive rate
        self.logger.info("True positive rate: " + str(metrics.recall_score(self.predictions['gt_label'], self.predictions['pred_label'])))

        # false positive rate
        self.logger.info("False positive rate: " + str(1 - metrics.recall_score(self.predictions['gt_label'], self.predictions['pred_label'])))

        # precision
        self.logger.info("Precision: " + str(metrics.precision_score(self.predictions['gt_label'], self.predictions['pred_label'])))

        # recall    
        self.logger.info("Recall: " + str(metrics.recall_score(self.predictions['gt_label'], self.predictions['pred_label'])))

    def roc_curve(self): 
        """
        This function generates an ROC curve for the given predictions. 
        """
        self.logger.info("Generating ROC curve")

        # toggle latex rendering
        plt.rc('text', usetex=True)

        # get true positive and false positive rates
        fpr, tpr, thresholds = metrics.roc_curve(self.predictions['gt_label'],self.predictions['pred_score'], pos_label=1)

        # get area under the curve
        auc = metrics.auc(fpr, tpr)

        # plot the roc curve
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)

        # plot the random line
        plt.plot([0, 1], [0, 1], 'k--')

        # set plot labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")

        # save figure
        plt.savefig("roc_curve.png", bbox_inches='tight')

        self.logger.info("ROC curve saved to roc_curve.png")
    



# test the class via __main__ 
if __name__ == "__main__":
    print("Testing AUC_ROC class")
    auc_roc = AUC_ROC("/share/ju/urbanECG/static/libraries/mmpretrain/results.pkl")
    auc_roc.inspect_predictions()
    auc_roc.roc_curve()
    print("Testing complete")

        

