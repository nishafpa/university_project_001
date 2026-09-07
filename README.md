# Project Title

UC Berkely Professional Certificate for ML &amp; AI - Required Assignment 17.1: Comparing Classifiers

## Description

This assignment is to compare various Classifier models such as KNN, Logistic Regression, Decision Trees and SVM (Support Vector Machine), in terms of the Performance such as Training Time, Test time, Hyperparameter Tuning time along with the Accuracy Score. The data used for this exercise is the direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit or not, based on various input features such as Bank Client data, Current and Overall Campaign Related data and Economic Indicator Related data.

## Getting Started

### Dependencies

* This Python assignment references the Libraries - pandas, matplotlib, seaborn, time, warnings and sklearn (scikit-learn)

### Executing program

* Click the below URL to Navigate to the Code, to Download and to Execute on any Python Notebook Cell by Cell from Top to Bottom
* [Classifier Models Comparison](https://github.com/nishafpa/university_project_001/blob/main/TermDepositPrediction.ipynb)

## Help

Brief Explanation and Analysis are provided below each cell, as Cell Output and Markdown

## Author

Nishaf P Abdul Majeed<BR>
Senior Data Architect/Modeler<BR>
[nishafpa@gmail.com](mailto:nishafpa@gmail.com)<BR>
[LinkedIn](https://www.linkedin.com/in/nishaf-abdul-majeed-04b13ba3/)

## Version History

* 0.1 - Initial Release

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments

UC Berkely Tutors for this Professional Course in Machine Learning and Artificial Intelligence

## Summary of the Findings

With the default parameters, below is the Summary of the Model Performance and Accuracy:
1) Elapsed Train Time - SVM Polynomial Kernel is the most time consuming to Train.
2) Train Accuracy - Decision Trees outperformed. Rest all models outperformed the baseline Dummy model and their Train Accuracy is more or less similar.
3) Test Accuracy - Decision Trees was great in Train Accuracy, it performed worst in Test Accuracy, even lower than the baseline Dummy model. Rest all models performed more or less similar leading by LinearSVC.

After the Hyperparameters tuning, below is the Summary of the Model Performance and Accuracy:
1) Elapsed Hyperparameter Tuning Time - Logistic Regression took the longest time to tune the Hyperparameters. SVM (Poly) is another slowest, which would have taken even longer time without 3GB Cache_size. LinearSVC is the fastest to tune the Hyperparameters.
2) Elapsed Train Time - SVM Polynomial Kernel and Logistic Regression are the most time consuming to Train. KNN is the fastest.
3) Train Accuracy - KNN outperformed. SVM (Poly) dropped its Train Accuracy a lot since the tuning parameters are limited due to the performance impact on the tuning time.
4) Test Accuracy - KNN outperformed in Test Accuracy as well. SVM (Poly) dropped its Test Accuracy a lot as well since the tuning parameters are limited due to the performance impact on the tuning time.