from gcornilib.DataManipulation import MLPrePostProcessing as dm2
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from functools import reduce


# Network Params
TV_PERC = 0.8
BATCH_SIZE = 128
EPOCHS = 10

# Features Params
feat_ft = ['Posizione', 'Corrente', 'Forza']
PAST_STEPS = 2
feat_steps = 1
feat_shape = (PAST_STEPS//feat_steps, len(feat_ft))

# Target Params
targ_ft = ['Posizione', 'Forza']
STEPS_AHEAD = 2
target_steps = 1
targ_shape = (STEPS_AHEAD//target_steps, len(targ_ft))


if __name__ == "__main__":

    # handle data range
    data_scaler = dm2.DataBalancer(min_max=True)

    # ETL training data
    print("> ETL pipeline...")

    (X_train, y_train), (X_valid, y_valid) = dm2.load_dataset_csv("./Data/gt.csv") \
        .pipe(dm2.drop_columns, ['Velocita', 'Accelerazione', 'StatoPLC', 'CodiceAllarme']) \
        .pipe(dm2.convert_data_to_type, 'float32') \
        .pipe(data_scaler.scale) \
        .pipe(dm2.series_to_supervised,
              n_in=PAST_STEPS,
              n_out=STEPS_AHEAD,
              features_filter=feat_ft,
              target_filter=targ_ft,
              features_steps=1,
              target_steps=1
              ) \
        .pipe(dm2.dataset_splitter,
              n_targets=reduce(np.multiply, targ_shape),
              tv_perc=TV_PERC
              )

    # # reshape X splits according to LSTM requirements
    X_train = X_train.reshape((-1, feat_shape[0], feat_shape[1]))
    X_valid = X_valid.reshape((-1, feat_shape[0], feat_shape[1]))

    print("\tTraining on {} samples".format(X_train.shape[0]))
    print("\tEvaluating on {} samples".format(X_valid.shape[0]))
    print("\tFeatures: {} / Targets: {}".format(reduce(np.multiply, feat_shape), y_train.shape[1]))
    print("> Done!")

    # TensorFlow DataSet
    print("> TensorFlow DataSet...")

    tf_dataset = tf.data.Dataset\
        .from_tensor_slices((X_train, y_train))\
        .batch(BATCH_SIZE)
    print("> Done!")

    # TensorFlow Model
    print("> TensorFlow Model...")

    model = tf.keras.Sequential(name="lstm_reg_1sa")
    model.add(tf.keras.layers.LSTM(
        50,
        input_shape=(X_train.shape[1], X_train.shape[2]),
        name="lstm_1_input")
    )
    model.add(tf.keras.layers.Dense(
        reduce(np.multiply, targ_shape),
        activation=tf.keras.activations.linear,
        name="regression_output")
    )
    model.compile(
        loss=tf.keras.losses.logcosh,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['mae', 'mse']
    )
    model.summary()

    print("> Done!")

    # Training phase
    print("> Start training...")

    for epoch in range(EPOCHS):
        for batch_features, targets in tf_dataset:
            train_loss, train_mae, train_mse = model.train_on_batch(batch_features, targets)

        print('Epoch #{}\t Loss: {:.6f}\tErrors: {:.6f} {:.6f}'.format(
            epoch + 1,
            train_loss,
            train_mae,
            train_mse)
        )

    print("> Done!")

    # Validation phase
    print("> Validating...")
    loss, mae, mse = model.evaluate(X_valid, y_valid)
    print("> Done!")

    # Plot predictions
    print("> Plotting Predictions...")

    # Put back data to their original range values
    original_target = data_scaler.rescale(y_valid, targ_ft)

    predictions = model.predict(X_valid)
    predicted_target = data_scaler.rescale(predictions, targ_ft)

    future_ts = 1  # t+0, t+1, t+2, ...
    variable = 1  # index of predicted variable to plot

    plt.figure()
    plt.title("{} at t+{}".format(targ_ft[variable], future_ts))

    real, = plt.plot(original_target[:, future_ts, variable], label='true values')
    pred, = plt.plot(predicted_target[:, future_ts, variable], label='predicted values')

    plt.legend(handles=[real, pred])
    plt.grid(True)
    plt.savefig('./Data/lstm_output.png', bbox_inches='tight')

    plt.show()

    print("> Done!")
