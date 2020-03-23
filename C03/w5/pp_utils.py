import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

##
def regression_model(n_cols, nodes_per_hlayer=[10,], opt='adam', loss='mean_squared_error'):
    """
    Define regression model as a python function
    """
    ## 1 - Create model
    model = Sequential()
    for ix, num_nodes in enumerate(nodes_per_hlayer):
      if ix == 0: # first layer
        model.add(Dense(num_nodes, activation='relu', input_shape=(n_cols,)))
      else:
        model.add(Dense(num_nodes, activation='relu'))
    model.add(Dense(1)) # output layer

    ## 2 - Compile model
    model.compile(optimizer=opt, loss=loss, metrics=[loss])
    return model

def split_data(df_pred, df_target, test_size=0.3, random_state=6776):
  X_train, X_test, y_train, y_test = train_test_split(df_pred, df_target,
                                                      test_size=test_size,
                                                      random_state=random_state)
  return (X_train, X_test, y_train, y_test) # tuple

def train_eval_loop(n_cols, df_pred, df_target, N=50, epochs=50, n_p_hl=[10]):
    mse_ary = []

    for ix in range(0, N):
        ## Reset model and therefore weigths
        model = regression_model(n_cols, nodes_per_hlayer=n_p_hl)

        ## Split the data into train and test set using opur wrapper function
        X_train, X_test, y_train, y_test = split_data(df_pred, df_target)

        ## Fit the model 
        _history = model.fit(X_train, y_train,
                             # validation_split=0.2, # Does not make sense for regression?
                             epochs=epochs,
                             verbose=0)

        ## Make Predictions
        pred = model.predict(X_test)

        ## Compare to ground truth - MSE not RMSE (so no np.sqrt)
        mse = mean_squared_error(y_test, pred)
        print("Iteration: {:2d} / MSE: {:1.5f}".format(ix, mse))

        ## Keep it for later
        mse_ary.append(mse)

    return mse_ary

def summary(mse_ary, label="baseline model"):
    print("Summary {:s}: ".format(label))
    np_ary = np.array(mse_ary, dtype=np.float64)

    ## NOTE: using unbiased std - which means diving by N-1 (where N is the size of sample here 50)
    ##       just in case, added biased std.
    print("mean(MSE): {:2.5f} / unbiased std(MSE): {:2.5f} / biased std(MSE): {:2.5f}"\
          .format(np.mean(np_ary), np.std(np_ary, ddof=1), np.std(np_ary, ddof=0)))
    return np_ary

def summary_ext(np_ary):
    ix_max = np_ary.argmax()
    ix_min = np_ary.argmin()

    print("max: {:2.5f} at epoch: {:2d} / min: {:2.5f} at epoch: {:2d}"\
          .format(np_ary[ix_max], ix_max, np_ary[ix_min], ix_min))
    return

def save_df(np_ary, label='mse_01_bl', fname='01_baseline_model.csv', prev_csv=None, prev_df=None):
    if prev_df is None and prev_csv is None:
        df = pd.DataFrame(np_ary, columns=[label])
    else:
       assert prev_csv is not None, "prev_csv should be defined along with prev_df"
       df = pd.read_csv(prev_csv)
       df[label] = list(np_ary)
    #
    assert fname is not None
    df.to_csv(fname, sep=',', encoding='utf-8', index=False)
    return df
