# Timeseries-Anomaly-Classifier
Done as part of Cybersecurity Group 4s project for DATA3001 2021.
Batch multi-feature time series anomaly detection using an LSTM, based on the paper.
Also some helper functions that support it

Just copy TimeSeriesErrorClassifier.py and start using the Classifier class.

1.Initialise a Classifier
2.run classifier.fitModel(some_data)
--by default, some_data must be the same length (i.e. same batch of devices,
 in the same order) as all future sets you'd want to predict later on
--otherwise, set "doWarmup=False" and then run classifier.warmUp(warmUpData)
 with a dataset with the same legnth as prediction inputs.
3.run enqueue(prediction_input)
4.the result should be a tensor with 0/1 showing whether or not they
are anomolous

FixedStepClassifier is a version of classifier that remembers a different
distribution for the errors of each timestep. It does this by assuming that
the training data used for fitting is representative of a complete cycle

When you are satisfied with a created model you can save it into a JSON using
myClassifier.save(filePath) and reload it using Classifier.load(filePath)

Even though it's supposed to be a capstone it's not all that impressive


REQUIREMENTS
tensorflow 2.7.0 and all associated dependencies
