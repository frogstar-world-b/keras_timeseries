# Fit a shallow nueral network with 1 input, 1 hidden layer with 8 neurons, and an output layer.
# A time series problem is converted into a regression problem by fitting:
#  y(t) to [y(t-1) ... y(t-lag)]

# Data:
# International airline passengers: monthly totals in thousands. Jan 49 – Dec 60
# https://datamarket.com/data/set/22u3/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import click

@click.command()
@click.option('--lag', default=1, type=int, help='number of time lags used as predictors')

def main(lag):
	# fix random seed for reproducibility
	np.random.seed(7)

	#################################  DATA PREPARATION  #########################################
	# load the ds
	df = pd.read_csv('data/international-airline-passengers.csv', 
		             usecols=[1], engine='python', skipfooter=3)
	ds = df.values
	ds = ds.astype('float32')
	assert lag < len(ds)

	# split into train and test sets
	train_size = int(len(ds) * 0.67)
	test_size = len(ds) - train_size
	train, test = ds[0:train_size,:], ds[train_size:len(ds),:]
	print(len(train), len(test))

	# convert an array of values into a ds matrix
	def create_ds(ds, lag=1):
		dataX, dataY = [], []
		for i in range(len(ds)-lag-1):
			a = ds[i:(i+lag), 0]
			dataX.append(a)
			dataY.append(ds[i + lag, 0])
		return np.array(dataX), np.array(dataY)

	# reshape into X=t and Y=t+1
	#lag = 10
	trainX, trainY = create_ds(train, lag)
	testX, testY = create_ds(test, lag)
	n_obs = len(trainX)

	##################################  FIT SHALLOW NN  ##########################################

	# The Sequential model is a linear stack of layers:
	# https://keras.io/getting-started/sequential-model-guide/
	model = Sequential()
	# hidden layer
	# The ReLU function is f(x)=max(0,x)
	# http://stats.stackexchange.com/questions/226923/why-do-we-use-relu-in-neural-networks-and-how-do-we-use-it
	model.add(Dense(8, input_dim=trainX.shape[1], activation='relu'))
	# output layer
	model.add(Dense(1))
	# compile
	# Adam: https://arxiv.org/abs/1412.6980v8
	model.compile(loss='mean_squared_error', optimizer='adam')
	# nb_epoch: total number of iterations on the data.
	# verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch
	# batch_size: number of samples per gradient update (each foward/backward propagation step)
	model.fit(trainX, trainY, nb_epoch=200, batch_size=2, verbose=1)

	##################################  EVALUATE/PLOT ############################################

	# Estimate model performance
	trainScore = model.evaluate(trainX, trainY, verbose=0)
	print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, np.sqrt(trainScore)))
	testScore = model.evaluate(testX, testY, verbose=0)
	print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, np.sqrt(testScore)))

	# generate predictions for training & test sets
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)

	# orrectly align train predictions for plotting
	trainPredictPlot = np.empty_like(ds)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[lag:len(trainPredict)+lag, :] = trainPredict

	# correctly align test predictions for plotting
	testPredictPlot = np.empty_like(ds)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(trainPredict)+(lag*2)+1:len(ds)-1, :] = testPredict

	# plot baseline and predictions
	plt.plot(ds, label = 'original dataset')
	plt.plot(trainPredictPlot, label = 'predictd training set')
	plt.plot(testPredictPlot, label = 'predicted test set')
	plt.ylabel('# of international airline passengers', fontsize=16)
	plt.xlabel('# of months from Jan 1949 – Dec 1960', fontsize=16)
	plt.title('Using the last {} lags as predictor(s)'.format(lag), fontsize=16)
	plt.legend(loc="upper left")
	plt.savefig('lag{}.png'.format(lag))
	plt.show()

if __name__ == "__main__":
	main()
