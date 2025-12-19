{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Contents\n",
    "\n",
    "* Logistic regression\n",
    "* Support Vector Machine\n",
    "* Bagging (Random Forest)\n",
    "* Boosting (XGBoost)\n",
    "* Neural Network (tensorflow/keras)\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data Source: https://www.kaggle.com/mlg-ulb/creditcardfraud "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Importing packages and data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
   },
   "outputs": [],
   "source": [
    "#importing packages\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "\n",
    "import os\n",
    "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "    for filename in filenames:\n",
    "        print(os.path.join(dirname, filename))\n",
    "        \n",
    "from IPython.core.display import HTML\n",
    "HTML(\"\"\"\n",
    "<style>\n",
    ".output_png {\n",
    "    display: table-cell;\n",
    "    text-align: center;\n",
    "    vertical-align: middle;\n",
    "}\n",
    "</style>\n",
    "\"\"\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#importing data from kaggle\n",
    "df = pd.read_csv(\"creditcard.csv\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data processing and undersampling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.drop(\"Time\", axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We need to standardize the 'Amount' feature before modelling. \n",
    "For that, we use the StandardScaler function from sklearn. Then, we just have to drop the old feature :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn import preprocessing\n",
    "scaler = preprocessing.StandardScaler()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#standard scaling\n",
    "df['std_Amount'] = scaler.fit_transform(df['Amount'].values.reshape (-1,1))\n",
    "\n",
    "#removing Amount\n",
    "df = df.drop(\"Amount\", axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now, let's have a look at the class :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.countplot(x=\"Class\", data=df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The dataset is highly imbalanced ! \n",
    "It's a big problem because classifiers will always predict the most common class without performing any analysis of the features and it will have a high accuracy rate, obviously not the correct one. To change that, I will proceed to random undersampling.  \n",
    "\n",
    "The simplest undersampling technique involves randomly selecting examples from the majority class and deleting them from the training dataset. This is referred to as random undersampling.\n",
    "\n",
    "Although simple and effective, a limitation of this technique is that examples are removed without any concern for how useful or important they might be in determining the decision boundary between the classes. This means it is possible, or even likely, that useful information will be deleted.\n",
    "\n",
    "### <center>How undersampling works :</center>\n",
    "<center><img src= \"https://miro.medium.com/max/335/1*YH_vPYQEDIW0JoUYMeLz_A.png\">\n",
    "\n",
    "\n",
    "\n",
    "To undersample, we can use the package imblearn with RandomUnderSampler function !"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import imblearn\n",
    "from imblearn.under_sampling import RandomUnderSampler \n",
    "\n",
    "undersample = RandomUnderSampler(sampling_strategy=0.5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "cols = df.columns.tolist()\n",
    "cols = [c for c in cols if c not in [\"Class\"]]\n",
    "target = \"Class\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#define X and Y\n",
    "X = df[cols]\n",
    "Y = df[target]\n",
    "\n",
    "#undersample\n",
    "X_under, Y_under = undersample.fit_resample(X, Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pandas import DataFrame\n",
    "test = pd.DataFrame(Y_under, columns = ['Class'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#visualizing undersampling results\n",
    "fig, axs = plt.subplots(ncols=2, figsize=(13,4.5))\n",
    "sns.countplot(x=\"Class\", data=df, ax=axs[0])\n",
    "sns.countplot(x=\"Class\", data=test, ax=axs[1])\n",
    "\n",
    "fig.suptitle(\"Class repartition before and after undersampling\")\n",
    "a1=fig.axes[0]\n",
    "a1.set_title(\"Before\")\n",
    "a2=fig.axes[1]\n",
    "a2.set_title(\"After\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_under, Y_under, test_size=0.2, random_state=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#importing packages for modeling\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from xgboost import XGBClassifier\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import layers\n",
    "from keras.layers import Dropout\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense\n",
    "from keras.layers import BatchNormalization\n",
    "\n",
    "from sklearn import metrics\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.metrics import roc_curve\n",
    "from sklearn.metrics import roc_auc_score\n",
    "from sklearn.metrics import auc\n",
    "from sklearn.metrics import precision_recall_curve"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#train the model\n",
    "model1 = LogisticRegression(random_state=2)\n",
    "logit = model1.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#predictions\n",
    "y_pred_logit = model1.predict(X_test) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#scores\n",
    "print(\"Accuracy Logit:\",metrics.accuracy_score(y_test, y_pred_logit))\n",
    "print(\"Precision Logit:\",metrics.precision_score(y_test, y_pred_logit))\n",
    "print(\"Recall Logit:\",metrics.recall_score(y_test, y_pred_logit))\n",
    "print(\"F1 Score Logit:\",metrics.f1_score(y_test, y_pred_logit))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#print CM\n",
    "matrix_logit = confusion_matrix(y_test, y_pred_logit)\n",
    "cm_logit = pd.DataFrame(matrix_logit, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])\n",
    "\n",
    "sns.heatmap(cm_logit, annot=True, cbar=None, cmap=\"Blues\", fmt = 'g')\n",
    "plt.title(\"Confusion Matrix Logit\"), plt.tight_layout()\n",
    "plt.ylabel(\"True Class\"), plt.xlabel(\"Predicted Class\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#AUC\n",
    "y_pred_logit_proba = model1.predict_proba(X_test)[::,1]\n",
    "fpr_logit, tpr_logit, _ = metrics.roc_curve(y_test,  y_pred_logit_proba)\n",
    "auc_logit = metrics.roc_auc_score(y_test, y_pred_logit_proba)\n",
    "print(\"AUC Logistic Regression :\", auc_logit)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#ROC\n",
    "plt.plot(fpr_logit,tpr_logit,label=\"Logistic Regression, auc={:.3f})\".format(auc_logit))\n",
    "plt.plot([0, 1], [0, 1], 'k--')\n",
    "plt.xlabel('False positive rate')\n",
    "plt.ylabel('True positive rate')\n",
    "plt.title('Logistic Regression ROC curve')\n",
    "plt.legend(loc=4)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "logit_precision, logit_recall, _ = precision_recall_curve(y_test, y_pred_logit_proba)\n",
    "no_skill = len(y_test[y_test==1]) / len(y_test)\n",
    "plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')\n",
    "plt.plot(logit_recall, logit_precision, color='orange', label='Logistic')\n",
    "plt.xlabel('Recall')\n",
    "plt.ylabel('Precision')\n",
    "plt.title('Precision-Recall curve')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Classification metrics for Logistic Regression (rounded down) :\n",
    "- Accuracy : 0.94\n",
    "- F1 score : 0.92\n",
    "- AUC : 0.96"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. Support Vector Machine"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#train the model\n",
    "model2 = SVC(probability=True, random_state=2)\n",
    "svm = model2.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#predictions\n",
    "y_pred_svm = model2.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#scores\n",
    "print(\"Accuracy SVM:\",metrics.accuracy_score(y_test, y_pred_svm))\n",
    "print(\"Precision SVM:\",metrics.precision_score(y_test, y_pred_svm))\n",
    "print(\"Recall SVM:\",metrics.recall_score(y_test, y_pred_svm))\n",
    "print(\"F1 Score SVM:\",metrics.f1_score(y_test, y_pred_svm))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#CM matrix\n",
    "matrix_svm = confusion_matrix(y_test, y_pred_svm)\n",
    "cm_svm = pd.DataFrame(matrix_svm, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])\n",
    "\n",
    "sns.heatmap(cm_svm, annot=True, cbar=None, cmap=\"Blues\", fmt = 'g')\n",
    "plt.title(\"Confusion Matrix SVM\"), plt.tight_layout()\n",
    "plt.ylabel(\"True Class\"), plt.xlabel(\"Predicted Class\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#AUC\n",
    "y_pred_svm_proba = model2.predict_proba(X_test)[::,1]\n",
    "fpr_svm, tpr_svm, _ = metrics.roc_curve(y_test,  y_pred_svm_proba)\n",
    "auc_svm = metrics.roc_auc_score(y_test, y_pred_svm_proba)\n",
    "print(\"AUC SVM :\", auc_svm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#ROC\n",
    "plt.plot(fpr_svm,tpr_svm,label=\"SVM, auc={:.3f})\".format(auc_svm))\n",
    "plt.plot([0, 1], [0, 1], 'k--')\n",
    "plt.xlabel('False positive rate')\n",
    "plt.ylabel('True positive rate')\n",
    "plt.title('SVM ROC curve')\n",
    "plt.legend(loc=4)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "svm_precision, svm_recall, _ = precision_recall_curve(y_test, y_pred_svm_proba)\n",
    "no_skill = len(y_test[y_test==1]) / len(y_test)\n",
    "plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')\n",
    "plt.plot(svm_recall, svm_precision, color='orange', label='SVM')\n",
    "plt.xlabel('Recall')\n",
    "plt.ylabel('Precision')\n",
    "plt.title('Precision-Recall curve')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Classification metrics for SVM (rounded down) :\n",
    "- Accuracy : 0.94\n",
    "- F1 score : 0.92\n",
    "- AUC : 0.97"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. Ensemble learning : Bagging (Random Forest)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#train the model\n",
    "model3 = RandomForestClassifier(random_state=2)\n",
    "rf = model3.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#predictions\n",
    "y_pred_rf = model3.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#scores\n",
    "print(\"Accuracy RF:\",metrics.accuracy_score(y_test, y_pred_rf))\n",
    "print(\"Precision RF:\",metrics.precision_score(y_test, y_pred_rf))\n",
    "print(\"Recall RF:\",metrics.recall_score(y_test, y_pred_rf))\n",
    "print(\"F1 Score RF:\",metrics.f1_score(y_test, y_pred_rf))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#CM matrix\n",
    "matrix_rf = confusion_matrix(y_test, y_pred_rf)\n",
    "cm_rf = pd.DataFrame(matrix_rf, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])\n",
    "\n",
    "sns.heatmap(cm_rf, annot=True, cbar=None, cmap=\"Blues\", fmt = 'g')\n",
    "plt.title(\"Confusion Matrix RF\"), plt.tight_layout()\n",
    "plt.ylabel(\"True Class\"), plt.xlabel(\"Predicted Class\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#AUC\n",
    "y_pred_rf_proba = model3.predict_proba(X_test)[::,1]\n",
    "fpr_rf, tpr_rf, _ = metrics.roc_curve(y_test,  y_pred_rf_proba)\n",
    "auc_rf = metrics.roc_auc_score(y_test, y_pred_rf_proba)\n",
    "print(\"AUC Random Forest :\", auc_rf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#ROC\n",
    "plt.plot(fpr_rf,tpr_rf,label=\"Random Forest, auc={:.3f})\".format(auc_rf))\n",
    "plt.plot([0, 1], [0, 1], 'k--')\n",
    "plt.xlabel('False positive rate')\n",
    "plt.ylabel('True positive rate')\n",
    "plt.title('Random Forest ROC curve')\n",
    "plt.legend(loc=4)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "rf_precision, rf_recall, _ = precision_recall_curve(y_test, y_pred_rf_proba)\n",
    "no_skill = len(y_test[y_test==1]) / len(y_test)\n",
    "plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')\n",
    "plt.plot(rf_recall, rf_precision, color='orange', label='RF')\n",
    "plt.xlabel('Recall')\n",
    "plt.ylabel('Precision')\n",
    "plt.title('Precision-Recall curve')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Classification metrics for Random Forest (rounded down) :\n",
    "- Accuracy : 0.95\n",
    "- F1 score : 0.93\n",
    "- AUC : 0.97"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 4. Ensemble learning : Boosting (XGBoost)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#train the model\n",
    "model4 = XGBClassifier(random_state=2)\n",
    "xgb = model4.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#predictions\n",
    "y_pred_xgb = model4.predict(X_test) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#scores\n",
    "print(\"Accuracy XGB:\",metrics.accuracy_score(y_test, y_pred_xgb))\n",
    "print(\"Precision XGB:\",metrics.precision_score(y_test, y_pred_xgb))\n",
    "print(\"Recall XGB:\",metrics.recall_score(y_test, y_pred_xgb))\n",
    "print(\"F1 Score XGB:\",metrics.f1_score(y_test, y_pred_xgb))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#CM matrix\n",
    "matrix_xgb = confusion_matrix(y_test, y_pred_xgb)\n",
    "cm_xgb = pd.DataFrame(matrix_xgb, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])\n",
    "\n",
    "sns.heatmap(cm_xgb, annot=True, cbar=None, cmap=\"Blues\", fmt = 'g')\n",
    "plt.title(\"Confusion Matrix XGBoost\"), plt.tight_layout()\n",
    "plt.ylabel(\"True Class\"), plt.xlabel(\"Predicted Class\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#AUC\n",
    "y_pred_xgb_proba = model4.predict_proba(X_test)[::,1]\n",
    "fpr_xgb, tpr_xgb, _ = metrics.roc_curve(y_test,  y_pred_xgb_proba)\n",
    "auc_xgb = metrics.roc_auc_score(y_test, y_pred_xgb_proba)\n",
    "print(\"AUC XGBoost :\", auc_xgb)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#ROC\n",
    "plt.plot(fpr_xgb,tpr_xgb,label=\"XGBoost, auc={:.3f})\".format(auc_xgb))\n",
    "plt.plot([0, 1], [0, 1], 'k--')\n",
    "plt.xlabel('False positive rate')\n",
    "plt.ylabel('True positive rate')\n",
    "plt.title('XGBoost ROC curve')\n",
    "plt.legend(loc=4)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, y_pred_xgb_proba)\n",
    "no_skill = len(y_test[y_test==1]) / len(y_test)\n",
    "plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')\n",
    "plt.plot(xgb_recall, xgb_precision, color='orange', label='XGB')\n",
    "plt.xlabel('Recall')\n",
    "plt.ylabel('Precision')\n",
    "plt.title('Precision-Recall curve')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Classification metrics for XGBoost (rounded down) :\n",
    "- Accuracy : 0.95\n",
    "- F1 score : 0.93\n",
    "- AUC : 0.97"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 5. Multi Layer Perceptron"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#train the model\n",
    "model5 = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(100,100), random_state=2)\n",
    "mlp = model5.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model5.get_params(deep=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#predictions\n",
    "y_pred_mlp = model5.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#scores\n",
    "print(\"Accuracy MLP:\",metrics.accuracy_score(y_test, y_pred_mlp))\n",
    "print(\"Precision MLP:\",metrics.precision_score(y_test, y_pred_mlp))\n",
    "print(\"Recall MLP:\",metrics.recall_score(y_test, y_pred_mlp))\n",
    "print(\"F1 Score MLP:\",metrics.f1_score(y_test, y_pred_mlp))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#CM matrix\n",
    "matrix_mlp = confusion_matrix(y_test, y_pred_mlp)\n",
    "cm_mlp = pd.DataFrame(matrix_mlp, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])\n",
    "\n",
    "sns.heatmap(cm_mlp, annot=True, cbar=None, cmap=\"Blues\", fmt = 'g')\n",
    "plt.title(\"Confusion Matrix MLP\"), plt.tight_layout()\n",
    "plt.ylabel(\"True Class\"), plt.xlabel(\"Predicted Class\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#AUC\n",
    "y_pred_mlp_proba = model5.predict_proba(X_test)[::,1]\n",
    "fpr_mlp, tpr_mlp, _ = metrics.roc_curve(y_test,  y_pred_mlp_proba)\n",
    "auc_mlp = metrics.roc_auc_score(y_test, y_pred_mlp_proba)\n",
    "print(\"AUC MLP :\", auc_mlp)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#ROC\n",
    "plt.plot(fpr_mlp,tpr_mlp,label=\"MLPC, auc={:.3f})\".format(auc_mlp))\n",
    "plt.plot([0, 1], [0, 1], 'k--')\n",
    "plt.xlabel('False positive rate')\n",
    "plt.ylabel('True positive rate')\n",
    "plt.title('Multilayer Perceptron ROC curve')\n",
    "plt.legend(loc=4)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mlp_precision, mlp_recall, _ = precision_recall_curve(y_test, y_pred_mlp_proba)\n",
    "no_skill = len(y_test[y_test==1]) / len(y_test)\n",
    "plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')\n",
    "plt.plot(mlp_recall, mlp_precision, color='orange', label='MLP')\n",
    "plt.xlabel('Recall')\n",
    "plt.ylabel('Precision')\n",
    "plt.title('Precision-Recall curve')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Classification metrics for Multi Layer Perceptron (rounded down) :\n",
    "- Accuracy : 0.95\n",
    "- F1 score : 0.94\n",
    "- AUC : 0.98"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 6. Multilayer Neural Network with Tensorflow/Keras"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#train the model\n",
    "model = Sequential()\n",
    "model.add(Dense(32, input_shape=(29,), activation='relu')),\n",
    "model.add(Dropout(0.2)),\n",
    "model.add(Dense(16, activation='relu')),\n",
    "model.add(Dropout(0.2)),\n",
    "model.add(Dense(8, activation='relu')),\n",
    "model.add(Dropout(0.2)),\n",
    "model.add(Dense(4, activation='relu')),\n",
    "model.add(Dropout(0.2)),\n",
    "model.add(Dense(1, activation='sigmoid'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "opt = tf.keras.optimizers.Adam(learning_rate=0.001) #optimizer\n",
    "\n",
    "model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy']) #metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=15, verbose=1,mode='auto', baseline=None, restore_best_weights=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "history = model.fit(X_train.values, y_train.values, epochs = 6, batch_size=5, validation_split = 0.15, verbose = 0,\n",
    "                    callbacks = [earlystopper])\n",
    "history_dict = history.history"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "loss_values = history_dict['loss']\n",
    "val_loss_values=history_dict['val_loss']\n",
    "plt.plot(loss_values,'b',label='training loss')\n",
    "plt.plot(val_loss_values,'r',label='val training loss')\n",
    "plt.legend()\n",
    "plt.xlabel(\"Epochs\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "accuracy_values = history_dict['accuracy']\n",
    "val_accuracy_values=history_dict['val_accuracy']\n",
    "plt.plot(val_accuracy_values,'-r',label='val_accuracy')\n",
    "plt.plot(accuracy_values,'-b',label='accuracy')\n",
    "plt.legend()\n",
    "plt.xlabel(\"Epochs\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#predictions\n",
    "y_pred_nn = model.predict_classes(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#scores\n",
    "print(\"Accuracy Neural Net:\",metrics.accuracy_score(y_test, y_pred_nn))\n",
    "print(\"Precision Neural Net:\",metrics.precision_score(y_test, y_pred_nn))\n",
    "print(\"Recall Neural Net:\",metrics.recall_score(y_test, y_pred_nn))\n",
    "print(\"F1 Score Neural Net:\",metrics.f1_score(y_test, y_pred_nn))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#CM matrix\n",
    "matrix_nn = confusion_matrix(y_test, y_pred_nn)\n",
    "cm_nn = pd.DataFrame(matrix_nn, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])\n",
    "\n",
    "sns.heatmap(cm_nn, annot=True, cbar=None, cmap=\"Blues\", fmt = 'g')\n",
    "plt.title(\"Confusion Matrix Neural Network\"), plt.tight_layout()\n",
    "plt.ylabel(\"True Class\"), plt.xlabel(\"Predicted Class\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#AUC\n",
    "y_pred_nn_proba = model.predict_proba(X_test)\n",
    "fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test,y_pred_nn_proba)\n",
    "auc_keras = auc(fpr_keras, tpr_keras)\n",
    "print('AUC Neural Net: ', auc_keras)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#ROC\n",
    "plt.figure(1)\n",
    "plt.plot([0, 1], [0, 1], 'k--')\n",
    "plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))\n",
    "plt.xlabel('False positive rate')\n",
    "plt.ylabel('True positive rate')\n",
    "plt.title('Neural Net ROC curve')\n",
    "plt.legend(loc='best')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "nn_precision, nn_recall, _ = precision_recall_curve(y_test, y_pred_nn_proba)\n",
    "no_skill = len(y_test[y_test==1]) / len(y_test)\n",
    "plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')\n",
    "plt.plot(nn_recall, nn_precision, color='orange', label='TF NN')\n",
    "plt.xlabel('Recall')\n",
    "plt.ylabel('Precision')\n",
    "plt.title('Precision-Recall curve')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Classification metrics for Neural Network (rounded down) :\n",
    "- Accuracy : 0.95\n",
    "- F1 score : 0.94\n",
    "- AUC : 0.98"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
