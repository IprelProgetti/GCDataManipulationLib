import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


#########
# Utils #
#########

sparse_columns = [
    "VariableName",
    "Type",
    "Time",
    "ValueBool",
    "ValueStr",
    "ValueInt",
    "ValueDec",
    "State",
    "Properties"
]

columns_mapping = {
    "ValueBool": "bool",
    "ValueDec": "float64",
    "ValueInt": "int64",
    "ValueStr": "object"
}


def is_list_of_strings(l):
    if l and isinstance(l, list):
        return all(isinstance(elem, str) for elem in l)
    else:
        return False


def cast_param_input(column_filter):
    if isinstance(column_filter, str):
        return [column_filter]
    elif is_list_of_strings(column_filter):
        return column_filter
    else:
        return []


#############
# Data Prep #
#############

# Reminders:
# To split "horizontally" a DataFrame: df.head(N) / df.tail(M)
# To split "vertically" a DataFrame: df[['col1', 'col2', ..., 'colN']]
# To split in train/valid sets:
# # from sklearn.model_selection import train_test_split
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=123)


def load_dataset_csv(filename):
    df = pd.read_csv(filename, header=0, index_col=0).infer_objects()  # use date column as index
    return df


def drop_columns(df, columns_to_drop):
    columns_to_drop = cast_param_input(columns_to_drop)
    # if no columns are specified, drop nothing
    # otherwise, drop the specified columns
    return df.drop(columns_to_drop, axis=1)


def convert_data_to_type(df, target_type, column_filter=None):
    column_filter = cast_param_input(column_filter)
    if column_filter:
        # if there are column filters, only convert the filtered columns
        for col in column_filter:
            df[col] = df[col].astype(target_type)
    else:
        # if no column filters are specified, convert the whole DataFrame
        df = df.astype(target_type)
    return df


# def encode_string_labels(df):
#     def do_encoding(values, str_columns):
#         encoder = LabelEncoder()
#         encoded = encoder.fit_transform(
#             values[:, str_columns].ravel()
#         )
#         values[:, str_columns] = encoded.reshape(-1, len(str_columns))
#         return values
#
#     # DataFrame columns
#     all_cols_names = list(df.columns.values)
#     # DataFrame string columns
#     str_cols_names = list(df.select_dtypes(include='object').columns.values)
#
#     # Do nothing if no string columns are found
#     if not str_cols_names:
#         return df
#
#     # DataFrame string columns indexes
#     str_indexes = [all_cols_names.index(str_col) for str_col in str_cols_names]
#     # New DataFrame values
#     encoded_values = do_encoding(df.values, str_indexes)
#     # Pack original columns with encoded values
#     df = pd.DataFrame(encoded_values, columns=all_cols_names)
#
#     return df


def series_to_supervised(
        df,
        n_in=1, n_out=1,
        features_filter=None, target_filter=None,
        features_steps=1, target_steps=1,
        drop_nan=True):
    # ASSUMPTION: steps back(n_in)/ahead(n_out) are referred to the data sampling interval
    # # i.e.: target_steps=5 aims to predict:
    # #         the next 5 minutes if data is sampled every minute
    # #         the next 5 hours if data is sampled every hour
    # #         the next 40 seconds if data is sampled every 8 second

    # Containers for target values
    # # cols:   List of (maybe shifted) DataFrames
    # # names:  List of strings
    cols, names = list(), list()

    # input sequence (t-n, ... t-1) ==> features columns
    features_filter = cast_param_input(features_filter)
    tdf = df.copy() if not features_filter else df[features_filter]

    for i in range(n_in, 0, -features_steps):
        cols.append(tdf.shift(i))
        names += ['{0}(t-{1})'.format(j, i) for j in list(tdf.columns.values)]

    # forecast sequence (t, t+1, ... t+n) ==> target columns
    target_filter = cast_param_input(target_filter)
    pdf = df.copy() if not target_filter else df[target_filter]
    # target_range = range(0, n_out) if target_steps == 1 else range(target_steps, n_out+1, target_steps)

    for i in range(0, n_out, target_steps):
        cols.append(pdf.shift(-i))
        if i == 0:
            names += ['{0}(t)'.format(j) for j in list(pdf.columns.values)]
        else:
            names += ['{0}(t+{1})'.format(j, i) for j in list(pdf.columns.values)]

    # put all the DataFrames in cols one close to the other, on the columns axis
    agg = pd.concat(cols, axis=1)
    # rename the aggregated columns
    agg.columns = names

    # drop rows with NaN values
    if drop_nan:
        agg.dropna(inplace=True)

    return agg


##############
# Finalizers #
##############


def dataset_splitter(df, n_targets=1, tv_perc=0.8):
    # "horizontal" split (row-based) - train vs valid
    n_train = int(df.shape[0] * tv_perc)
    n_valid = int(df.shape[0] - n_train)

    df_train, df_valid = df.head(n_train), df.tail(n_valid)

    # "vertical" split (column-based) - features vs targets
    X_train, y_train = df_train[df_train.columns[:-n_targets]], df_train[df_train.columns[-n_targets:]]
    X_valid, y_valid = df_valid[df_valid.columns[:-n_targets]], df_valid[df_valid.columns[-n_targets:]]

    # values extractions
    X_train, X_valid = X_train.values, X_valid.values
    y_train, y_valid = y_train.values, y_valid.values

    return (X_train, y_train), (X_valid, y_valid)


##################
# Data balancing #
##################


class LabelHandler(object):
    def __init__(self):
        self._encoder = LabelEncoder()

    def encode_string_labels(self, df):
        def do_encoding(values):
            original_shape = values.shape
            encoded = self._encoder.fit_transform(values.ravel()).reshape(original_shape)

            return encoded

        # Get DataFrame string columns
        str_cols_names = list(df.select_dtypes(include='object').columns.values)

        # Do nothing if no string columns are found
        if not str_cols_names:
            return df

        # Encode DataFrame string values
        encoded_values = do_encoding(df[str_cols_names].values)

        # Replace string values with their encoding
        df[str_cols_names] = encoded_values

        return df

    def decode_string_labels(self, predictions):
        original_shape = predictions.shape
        decoded_data = self._encoder.inverse_transform(predictions.ravel()).reshape(original_shape)

        return decoded_data


class DataBalancer(object):
    def __init__(self, min_max=False):
        self._scaler = MinMaxScaler() if min_max else StandardScaler()
        self._helpers = None

    def scale(self, df):
        df_values = df.values
        df_columns = list(df.columns.values)

        scaled_data = self._scaler.fit_transform(df_values)
        scaled_df = pd.DataFrame(scaled_data, columns=df_columns)

        # Must be a DataFrame to exploit columns position information
        self._helpers = scaled_df.copy()

        return scaled_df

    def rescale(self, predictions, columns):
        # ASSUMPTION: DataSet rows are always >= than batched prediction rows

        # Make sure scaling has been done before
        if self._helpers is None:
            raise AssertionError(
                "You should have scaled your data before training. Please scale data and retrain your model first"
            )

        # Save original number of predictions
        N = predictions.shape[0]

        # Stack future time step predictions of the same column vertically, one below the other
        predictions = predictions.reshape(-1, len(columns))

        # # Now predictions is a numpy array with one column for each predicted variable
        # # Predicted time steps are collapsed within each feature column according to the same order

        # Fetch helper values for rescaling, as many rows as needed (nb_prediction_rows * nb_future_time_steps)
        helper_df = self._helpers.head(predictions.shape[0]).copy()

        # Put scaled predictions within the proper columns
        helper_df[columns] = predictions

        # put predictions in the first NxM sub-matrix of the helper
        # self._helpers[:predictions.shape[0], -predictions.shape[1]:] = predictions
        # self._helpers[columns] = predictions
        # helper_values = self._helpers[columns].values
        # helper_values[:predictions.shape[0], :] = predictions

        # rescale the whole helper structure (to match the scaler data expected size)
        rescaled_struct = self._scaler.inverse_transform(
            helper_df.values
        )

        helper_df[helper_df.columns.values] = rescaled_struct

        # return only the first NxM sub-matrix of the helper (rescaled predictions)
        rescaled_data = helper_df[columns].values

        return rescaled_data.reshape(N, -1, len(columns))


##########################
# Data format conversion #
##########################


def sparse_to_table(df):
    # group by Time (== index) to make sure you have one table row for each specific time instance
    # set column names to VariableNames (all variable values for each time instance must be put in the same row)
    # each variable value, for each time instance, is the unique value among the four sparse columns (3 None + 1 value)
    table_df = pd.pivot_table(
        df,
        index=sparse_columns[2],
        columns=sparse_columns[0],
        values=[sparse_columns[3], sparse_columns[4], sparse_columns[5], sparse_columns[6]],
        aggfunc=np.unique
    )

    # type DataFrame columns accordingly
    for typed_column in table_df.columns.values:
        table_df[typed_column] = table_df[typed_column].astype(columns_mapping[typed_column[0]])

    # rename columns to match the distinct variable names
    table_df.columns = table_df.columns.map('{0[1]}'.format)

    # remove the index name
    table_df.index.name = None

    return table_df


def table_to_sparse(df, time_column_name="Time"):
    def all_rows():
        yield from df.iterrows()

    def to_sparse_encoding(idx, row):
        def type_to_str(var_type):
            return str(var_type).split("'")[1]

        # Convert pandas Series into dict
        row = row.to_dict()

        # Sparse encoding: one line per variable
        # # If df has a time column, ignore it and embed time values within the sparse encoding
        # # Otherwise, use the index value - convert it to datetime only if it is not an instance of int
        tse = [{
            sparse_columns[0]: var_name,
            sparse_columns[1]: type_to_str(type(row[var_name])),
            sparse_columns[2]: row[time_column_name] if time_column_name in row.keys() else idx,
            sparse_columns[3]: row[var_name] if isinstance(row[var_name], bool) else None,
            sparse_columns[4]: row[var_name] if isinstance(row[var_name], str) else None,
            sparse_columns[5]: row[var_name] if not isinstance(row[var_name], bool) and isinstance(row[var_name], int) else None,
            sparse_columns[6]: row[var_name] if isinstance(row[var_name], float) else None,
            sparse_columns[7]: None,
            sparse_columns[8]: None
        } for var_name in row.keys() if var_name != time_column_name]

        return tse

    # List of rows, each row being a list of variables
    list_df = [to_sparse_encoding(i, r) for i, r in all_rows()]

    # List of sparse-encoded variables, for each row of the original DataFrame, and for each variable of each row
    list_var = [variable for row in list_df for variable in row]

    # Reframe the sparse-encoded variables into a DataFrame
    sparse_df = pd.DataFrame(list_var, columns=sparse_columns).infer_objects()
    sparse_df = sparse_df.where((pd.notnull(sparse_df)), None)

    # Set the time column (the 3rd in the sparse format) to datetime type
    if sparse_df[sparse_columns[2]].dtype != 'int':
        sparse_df[sparse_columns[2]] = pd.to_datetime(sparse_df[sparse_columns[2]])

    return sparse_df
