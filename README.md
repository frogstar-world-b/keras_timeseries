# keras_timeseries

## Summary
In this exercise, we recreate an example by Jason Brownlee. Somethings were modified:

1. The docstrings have been expanded to provide explanations (including relevant URLs) of the arguments to the keras functions.
2. The file is now intended to run using a terminal command incorporating the click package; http://click.pocoo.org/.
3. Once the keras model is run, a figure will be displayed, and a png-format image will be saved to the current working directly.

## Running the code
To run this file using the last time lag `y(t-1)` as a predictor for `y(t)`, type into your terminal command line:
```
python keras_nn_timeseries1.py --lag 1
```
You will obtian the following plot:
<img src="https://github.com/frogstar-world-b/keras_timeseries/blob/master/lag1.png" width="400">

To use the last 10 time lags `y(t-1), ... y(t-10)` as a predictors for `y(t)`, type into your terminal command line:
```
python keras_nn_timeseries1.py --lag 10
```
You will obtian the following plot:
<img src="https://github.com/frogstar-world-b/keras_timeseries/blob/master/lag10.png" width="400">

For help, type into your terminal command line:
```
python keras_nn_timeseries1.py --help
```

## References
* http://machinelearningmastery.com/ 
* https://datamarket.com/data/set/22u3/
