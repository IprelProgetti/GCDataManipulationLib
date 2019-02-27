from gcornilib import DataManipulationV2 as dm2
import pandas as pd
import numpy as np


if __name__ == "__main__":

    #
    # setup your data preprocessing pipeline
    #
    df = dm2.load_dataset_csv("/users/cornigab/Documents/jptnotebooks/gt.csv")\
        .pipe(dm2.drop_columns, ['StatoPLC', 'Posizione'])\
        .pipe(dm2.encode_string_labels)\
        .pipe(dm2.convert_data_to_type, 'float32')\
        .pipe(dm2.convert_data_to_type, 'int32', 'CodiceAllarme')\
        .pipe(dm2.series_to_supervised, n_in=1, n_out=3, target_filter='Velocita')\
        .head()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

    #
    # convert narrow table representations to wide ones and vice versa
    #
    df1 = pd.DataFrame([[1, 2.0, "3", True], [5, 6.0, "7", False]], columns="a b c d".split()).infer_objects()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df1.dtypes)
        print(df1)

    df2 = dm2.table_to_sparse(df1)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df2.dtypes)
        print(df2)

    df3 = dm2.sparse_to_table(df2)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df3.dtypes)
        print(df3)

    #
    # Scale your input features within a given interval or mean/std, and back to their original representations
    #
    sc = dm2.DataBalancer(min_max=True)  # min_max=False to get a StandardScaler to enforce the same probability distr.

    training_data = np.array([[1., 2., 2.5, 7.8], [1., 22., 2.53, 7.68], [11., 2., 2.5, 76.8], [1., 222., 2.5, 7.8]])
    df4 = pd.DataFrame(training_data, columns="a b c d".split())
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df4)

    df5 = sc.scale(df4)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df5)

    # Do some predictions
    predictions = np.array([[5.], [2.5]])

    rsd = sc.rescale(predictions)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(pd.DataFrame(rsd, columns="a ".split()))
