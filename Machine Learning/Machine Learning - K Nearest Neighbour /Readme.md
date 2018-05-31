
## K-Nearest Neighbour

Experimented with unsupervised learning where lables of the data are unknown. Used KNN to build a model for classfied data to predict target class for new data points based on features.

Imported data and explored it a bit to get the feel of it. As the KNN classifier predicts the class of a given test observation by identifying the observations that are nearest to it, the scale of the variables matters. Any variables that are on a large scale will have a much larger effect on the distance between the observations, and hence on the KNN classifier, than variables that are on a small scale. So standardzing the variables becomes a must. 

Model is built starting with 1 neighbour and further improved with elbow method which suggets k=23 for optimum performance.
The model is evaluated by classification report and confusion matrix with 95% F1 and recall score.


