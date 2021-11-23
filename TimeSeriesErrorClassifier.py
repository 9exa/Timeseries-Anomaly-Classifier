import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import chi2
import json
#beyond preprocessing, everything the user needs should be in here

def makeWindow(data, window = 1):
    data = np.array(data)
    t = data.shape[-2]
    f = data.shape[-1]
    frontSize = data.shape[:-2]
    inds = (np.arange(window) +
            np.arange(t-window+1).reshape(t-window+1, 1))
    return data[:, inds].reshape(frontSize +(t-window+1, f*window))

#expects 3-dim
def getGaussian(errors, flatten = False):
    #flattening brings timesteps into batch dim, so we only get 1 distribution
    if flatten:
        errors = np.array(errors).reshape((-1, errors.shape[-1]))
        mean = np.mean(errors, axis = 0)
        return (
            mean, np.array(np.cov(errors, rowvar = False))
        )

    mean = np.mean(errors, axis = 0)
    errors = errors.transpose(1,0,2)
    return (
        mean, np.array([np.cov(e, rowvar = False) for e in errors])
    )
# #A neural net with a stateful LSTM
# def Predictor(outSize, stateful = False, bis = None, **kwargs):
#     # a stat
#     model = tf.keras.Sequential(**kwargs)
#     model.add(layers.LSTM(40, stateful = stateful,
#                 return_sequences=True, dropout = 0.2,
#                 # batch_input_shape = bis)
#                 ))
#     model.add(layers.Dropout(0.3))
#     model.add(layers.Dense(outSize))
#     model.add(layers.BatchNormalization())
#     model.compile(tf.optimizers.Adam(), tf.losses.MSE)
#     return model
#
# #not using stateful LSTM, batching is weird. Instead, we'll use fucntional API
# def Predictor(inShape, outSize, stateful = False):
#     inp = layers.Input(inShape)
#     x, h, c = layers.LSTM(40, return_sequences = True,
#                         dropout = 0.2, return_state = True)(inp)
#     x = layers.Dropout(0.3)(x)
#     x = layers.Dense(outSize)(x)
#     x = layers.BatchNormalization()(x)
#
#     if stateful:
#         model = tf.keras.Model(inputs=inp, outputs = (x, h, c))
#     else:
#         model = tf.keras.Model(inputs=inp, outputs = x)
#         model.compile(tf.optimizers.Adam(), tf.losses.MSE)
#     return model

class Predictor(tf.keras.Model):
    def __init__(self, outSize, **kwargs):
        super(Predictor, self).__init__()
        self.lstm = layers.LSTM(40, return_sequences = True,
                        dropout = 0.2, return_state = True)
        self.drop = layers.Dropout(0.3)
        self.dense = layers.Dense(outSize)
        self.batch = layers.BatchNormalization()
        self.compile(loss = "mse", optimizer="adam")
        # self.inpSize = inp
    def call(self, inputs, initialStates = None, training = False):

        x, b, c = self.lstm(inputs, initial_state = initialStates,
                            training = training)
        x = self.drop(x, training = training)
        x = self.dense(x, training = training)
        return (self.batch(x, training = training), b, c)
    def predict(self, inputs):
        x, h, c = self.call(inputs)
        return x
    #copied from tensorflow docs
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        with tf.GradientTape() as tape:
            y_pred, h , c = self.call(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred,
                                    regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


#a distance metric that, when normality holds, is chi2
def distStat(values, mean, invcov):
    #flatten the values if necessary
    # print(mean)
    # if len(values.shape) > len(mean.shape) + 1:
    #     values = tf.reshape(values, (-1, mean.shape[-1]))
    diff = values - mean
    #have to add a single dim to the end so that dot product works
    #also invcov may be very small, so use doubles for it
    diff = tf.cast(tf.expand_dims(diff, -1), "float64")
    return tf.cast(
                tf.matmul(diff, invcov @ diff, transpose_a = True),
                "float32")

class Classifier:
    """
    LSTM classifier:
    designed for real time predictions, so you can put in new observations
    and it should be able to output whether or not we think its an anomoly.

    Usage:
    1.Initialise a Classifier
    2.run classifier.fitModel(some_data)
    --by default, some_data must be the same length (i.e. same batch of devices,
     in the same order) as all future sets you'd want to predict later on
    --otherwise, set "doWarmup=False" and then run classifier.warmUp(warmUpData)
     with a dataset with the same legnth as prediction inputs.
    3.run enqueue(prediction_input)
    4.the result should be a tensor with 0/1 showing whether or not they
    are anomolous

    --setting useQuantiles=True will make the threshold determined by
      quantiles of the distance test stat as opposed to ppf chi2
    """
    def __init__(self, sigLevel=0.05, windowsize = 1, useQuantiles = False):
        #start by checking value untegrity
        if windowsize < 1:
            raise ValueError("windowsize cannot be less than 1")
        if sigLevel < 0 or 1 < sigLevel:
            raise ValueError("Significance Level must be between 0 and 1")
        self.built = False
        self.fitted = False
        self.windowsize = windowsize
        self.useQuantiles = useQuantiles
        self.sigLevel = sigLevel
        self.model = None
    #raises exceptions if trying to run fit with invalid arguments
    def errorHandellingFit(self, trainData, valData):
        if len(trainData.shape) != 3:
            raise ValueError("Expected Time Series (i.e.dim=3) for train data")
        elif valData != None and valData.shape != trainData.shape:
            raise ValueError(f"""Validation set {valData.shape} does
                    not match Training set shape {trainData.shape}""")
        elif trainData.shape[1] < self.windowsize +1:
            raise ValueError(f"""length of sequence ({trainData.shape[1]})
                    should be longer than Window Size ({self.windowsize})""")
    #estimates gaussian of errors and stores them. Overridden in subclasses
    def storeDistribution(self, errors):
        self.mean, cov = getGaussian(errors, flatten = True)

        self.invcov = tf.linalg.pinv(cov).numpy()
    def setThreshold(self, errors):
        if self.useQuantiles:
            d2 = distStat(errors, self.mean, self.invcov)
            self.threshold = np.quantile(d2, 1-self.sigLevel, axis = None)
        else:
            self.threshold = chi2.ppf(1-self.sigLevel, nFeatures)

    def fitModel(self, trainData, valData=None, doWarmup = True, **kwargs):
        """
        valData is used to determin gaussian/quantiles and the threshold
        If not supplied, it trainData will be gotten from a split trainData
        Expects time series to not already be windowed
        If do warmup is true, then also uses the input data to initialise the
        LSTM modelStates

        This class of Classifier will store 1 mean and 1 covariance for
        """
        #integrity checking
        self.errorHandellingFit(trainData, valData)

        trainDataOG = trainData
        if valData == None:
            trainData, valData = train_test_split(trainData, test_size = 0.3)
        #fitten' model
        inputs = makeWindow(trainData[:, :-1,:], self.windowsize)
        labels = trainData[:, self.windowsize:,:]
        nFeatures = trainData.shape[-1]
        self.inputShape = inputs.shape

        self.model = Predictor(nFeatures)
        self.model.fit(inputs, labels, **kwargs)
        valInp = makeWindow(valData[:, :-1, :], self.windowsize)
        valLab = valData[:, self.windowsize:, :]

        preds= self.model.predict(valInp)
        errors = valLab - preds

        self.storeDistribution(errors)

        self.setThreshold(errors)

        self.fitted = True
        #in the case of refitting, we need to to warm up the model again
        self.built = False
        #stor weights
        # self.tempWeights = tempModel.get_weights()
        if doWarmup:
            self.warmUp(trainDataOG)

    #runs data through a stated model, setting it up for future predictions
    #for data with that shape
    def warmUp(self, inputs):
        outputFeats = inputs.shape[-1]
        winInputs = makeWindow(inputs, self.windowsize)
        # # batches = min(inputs.shape[0], 32)
        # # batchShape = (batches,)+inputs.shape[1:-1]+(outputFeats,)
        #initialize the predictor
        # del self.model
        # self.model = Predictor(outputFeats)
        # self.model.build(batchShape)
        drop, self.hidden, self.context = self.model.call(winInputs)
        # self.model.set_weights(self.tempWeights)
        # del self.tempWeights
        self.last = inputs[:,-self.windowsize:,:]
        self.built = True

    def resetModelStates(self):
        #zeroes the internal nodel states
        self.model.reset_states()
    #saves internals as a json
    def save(self, modelFile):
        if not self.built:
            raise Error("Model must be fitted and warmed up "+
                    "before it can be saved")
        #for saving array of arrays or list of arrays into a json friendly form
        def deepList(item):
          try:
              it = iter(item)
          except TypeError:
              return float(item)
          return [deepList(thing) for thing in it]

        with open(modelFile, "w") as f:
            json.dump(
                    (
                        self.inputShape,
                        self.sigLevel,
                        self.windowsize,
                        self.useQuantiles,
                        deepList(self.last),
                        deepList(self.model.get_weights()),
                        deepList(self.hidden),
                        deepList(self.context),
                        deepList(self.mean),
                        deepList(self.invcov),
                        deepList(self.threshold)
                    ),
                    f)
    #loads Classifier from file
    def load(modelFile):
        self = Classifier()
        self.loadBase(modelFile)
        return self
    #part that doesn't get changed in subclassing.Probably should rewrite it
    #but I'm lazy
    def loadBase(self, modelFile):
        with open(modelFile, "r") as f:
            (
            self.inputShape,
            self.sigLevel,
            self.windowsize,
            self.useQuantiles,
            self.last,
            weights,
            self.hidden,
            self.context,
            self.mean,
            self.invcov,
            self.threshold
            ) = json.load(f)
        self.last = np.array(self.last)
        self.mean = np.array(self.mean)
        self.invcov = np.array(self.invcov)
        p = self.mean.shape[-1]
        self.model = Predictor(p)
        self.model(tf.ones(self.inputShape))
        weights = list(map(tf.Variable, weights))
        self.model.set_weights(weights)

        self.fitted, self.built = True, True

    def errorHandellingQueue(self, newdata):
        #newdata may either be an aarray or single input
        if not self.fitted:
            raise RuntimeError("Model not yet fitted. Need to run "
                                + "Classifier.fit")
        elif not self.built:
            raise RuntimeError("Model needs to be warmed up. Try running "
                    + "Classifiers.warmUp using the first few time steps of "
                    + "input/the training data.")
        elif len(newdata.shape) not in [2,3]:
            raise ValueError("Input must be a 2 or 3 dimensional array")
    def enqueue(self, newdata):
        self.errorHandellingQueue(newdata)

        data = np.concatenate([self.last, newdata], axis = 1)
        inputs = makeWindow(data[:, :-1,:], self.windowsize)
        labels = data[:,self.windowsize:,:]
        inStates = [tf.constant(self.hidden), tf.constant(self.context)]
        preds, self.hidden, self.context = self.model(inputs, inStates, False)
        errors = labels-preds

        self.last = data[:, -self.windowsize:, :]

        d2 = distStat(errors, self.mean, self.invcov)
        return np.where(d2 > self.threshold, 1, 0).reshape(d2.shape[:2])

    pass
#Like the classifier byt
class FixedStepClassifier(Classifier):
    """FixedStepClassifier will keep unique distribution
    for each timeStep. i.e the distribution of observations identical for a
    given timestep at each period.
    As such fitModel expects a exactly (period + windowsize) of the
    target data to train on.
    """
    def fitModel(self, trainData, valData=None, doWarmup = True, **kwargs):
        #since traindata is a full period, we'll loop back around by windowsize
        #in order to get initial states + correct num of distribution
        trainData = np.concatenate(
                    (trainData[:,-self.windowsize:,:], trainData),axis = 1
                    )
        super(FixedStepClassifier, self).fitModel(
             trainData, valData, doWarmup, **kwargs
        )
        #classifier saves how long the desired timestep it
        self.timeLength = trainData.shape[1] - self.windowsize
        #also stores the next timestep to be predicted
        self.timeStep = 0

    def storeDistribution(self, errors):
        self.mean, cov = getGaussian(errors, flatten = False)
        self.invcov = tf.linalg.pinv(cov).numpy()
    #same as classifier but without dimensiuon flatening
    def setThreshold(self, errors):
        if self.useQuantiles:
            d2 = distStat(errors, self.mean, self.invcov)
            self.threshold = np.quantile(d2, 1-self.sigLevel, axis = 0)
        else:
            self.threshold = chi2.ppf(1-self.sigLevel, nFeatures)
    def load(modelFile):
        self = FixedStepClassifier()
        self.loadBase(modelFile)
        return self

    def enqueue(self, newdata):
        #integ test
        self.errorHandellingQueue(newdata)

        #choose the cirresponding mean and variances to match the new data
        numObs = len(newdata)
        indexes = (np.arange(numObs) + self.timeStep) % self.timeLength
        self.timeStep = (self.timeStep + numObs) % self.timeLength

        mean = self.mean[indexes]
        invcov = self.invcov[indexes]
        #process as normal
        data = np.concatenate([self.last, newdata], axis = 1)
        inputs = makeWindow(data[:, :-1,:], self.windowsize)
        labels = data[:,self.windowsize:,:]
        inStates = [tf.constant(self.hidden), tf.constant(self.context)]
        preds, self.hidden, self.context = self.model(inputs, inStates, False)
        errors = labels-preds
        self.last = data[:, -self.windowsize:, :]

        d2 = distStat(errors, self.mean, self.invcov)
        return np.where(d2 > self.threshold, 1, 0).reshape(d2.shape[:2])


if __name__ == "__main__":
    model = Predictor(22)
    print(model)
