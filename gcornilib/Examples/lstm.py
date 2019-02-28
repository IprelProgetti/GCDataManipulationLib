from gcornilib.DataManipulation import MLPrePostProcessing as dm2
from matplotlib import pyplot as plt
import tensorflow as tf


STEPS_AHEAD = 1
PAST_STEPS = 1
TV_PERC = 0.8
BATCH_SIZE = 128
EPOCHS = 10


if __name__ == "__main__":
    # handle data range
    data_scaler = dm2.DataBalancer(min_max=True)

    # ETL training data
    print("> ETL pipeline...")

    (X_train, y_train), (X_valid, y_valid) = dm2.load_dataset_csv("./Data/gt.csv")\
        .pipe(dm2.drop_columns, ['StatoPLC', 'CodiceAllarme'])\
        .pipe(dm2.convert_data_to_type, 'float32')\
        .pipe(dm2.series_to_supervised, n_in=PAST_STEPS, n_out=STEPS_AHEAD, target_filter='Forza')\
        .pipe(data_scaler.scale)\
        .pipe(dm2.dataset_splitter, n_targets=STEPS_AHEAD, tv_perc=TV_PERC)

    # # reshape X splits according to LSTM requirements
    X_train = X_train.reshape((X_train.shape[0], PAST_STEPS, X_train.shape[1]))
    X_valid = X_valid.reshape((X_valid.shape[0], PAST_STEPS, X_valid.shape[1]))

    print("\tTraining on {} samples".format(X_train.shape[0]))
    print("\tEvaluating on {} samples".format(X_valid.shape[0]))
    print("\tFeatures: {} / Targets: {}".format(X_train.shape[2], y_train.shape[1]))
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
    model.add(tf.keras.layers.LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), name="lstm_1_input"))
    model.add(tf.keras.layers.Dense(STEPS_AHEAD, activation=tf.keras.activations.linear, name="regression_output"))

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

        print('Epoch #{}\t Loss: {:.6f}\tErrors: {:.6f} {:.6f}'.format(epoch + 1, train_loss, train_mae, train_mse))

    print("> Done!")

    # Validation phase
    print("> Validating...")
    loss, mae, mse = model.evaluate(X_valid, y_valid)
    print("> Done!")

    # Plot predictions
    print("> Plotting Predictions...")

    # Put back data to their original range values
    original_target = data_scaler.rescale(y_valid)

    predictions = model.predict(X_valid)
    predicted_target = data_scaler.rescale(predictions)

    plt.figure()
    plt.title("Predictions")

    real, = plt.plot(original_target, label='true values')
    pred, = plt.plot(predicted_target, label='predicted values')

    plt.legend(handles=[real, pred])
    plt.grid(True)

    plt.show()

    print("> Done!")
