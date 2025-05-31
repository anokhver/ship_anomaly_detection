### Labelling methodology
After the data has been cleaned, the anomalies in a learning set of data must be labeled. For manual labeling, python libraries already included (numpy, pandas) will be used.
The anomalies will be manually labeled according to a set of rules previosly discussed:

Long stop in the middle of the sea (more than certain distance from harbor - assuming that close proximity to harbor and other vessels can potentially impact this) 
Frequent or drastical speed change (Over X knots, while more than certain distance from harbor - assuming that close proximity to harbor and other vessels can potentially impact this) 
Frequent or drastical draught change
Starting and Ending point slightly differ from usual values of startPort and endPort.


Latitude and longitude that are more than certain distance from usual coords on the same trip 
route.
Frequent or drastical course change (Over X degrees on this Latitude/Longitude in comparison 
to other examples/trips)

The last two types of anomalies can be additionally pre-labeled with the use of unsupervised learning models from scikit-learn python library (potentially Isolation Forest, One-Class SVM, Local Outlier Factor). Then the labels will have to be manually evaluated.

It would be a good idea to put the data in a database file (sqlite), since it would make the verification and data manipulation easier. (Piotr)


### Result analysis
For result analysis, we can use both automatic and manual verification. Since we're already using the scikitlearn python library, we can use its functionality to compare the models (scikitlearn cross-validate) with the help of the test set. Manual verification would consists of verifying if results are in range of what we consider anomaly (checking with pandas code or database queries) and looking at the results visualised on diagrams. This can be done either with python matlplotlib or with feeding the data directly to the frontend of the application, depending on how developed it is by the time we have results.