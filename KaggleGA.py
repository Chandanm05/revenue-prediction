import shutil

import pandas as pd
from pandas.io.json import json_normalize
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.feature_column.feature_column import indicator_column
from tensorflow.python.feature_column.feature_column import categorical_column_with_hash_bucket
from tensorflow.python.framework import dtypes


def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(csv_path,
                     converters={column: json.loads for column in JSON_COLUMNS},
                     dtype={'fullVisitorId': 'str'},  # Important!!
                     nrows=nrows)

    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [str.format("{}.{}",column,subcolumn) for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print("Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


def load_chunked_df(df):
    df = df.reset_index(drop = True)
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [str.format("{}.{}",column,subcolumn) for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print("Shape = ", df.shape)
    return df


target = 'totals.transactionRevenue'
def getLabels():
    labels = []
    # After first training merge corelated features, remove redundent features
    labels.extend(['fullVisitorId',
                   'channelGrouping', 'socialEngagementType',
                   'device.browser', 'device.deviceCategory', 'device.operatingSystem',
                   'geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country', 'geoNetwork.metro', 'geoNetwork.region', 'geoNetwork.subContinent',
                   'totals.bounces','totals.hits', 'totals.newVisits', 'totals.pageviews', 'totals.visits',
                   'trafficSource.adContent', 'trafficSource.medium', 'trafficSource.referralPath', 'trafficSource.source',
                   ])
    return labels


print('labels = ',len(getLabels()))


def change_to_numeric(df, list):
    for col in list:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df


def feature_preprocessing(df):
    # Change numeric column datatype
    df = change_to_numeric(df, ['totals.hits', 'totals.pageviews', 'totals.visits'])
    # Fill string na  with ''
    df = df.fillna(value='')
    return df


def target_preprocessing(df):
    t = pd.to_numeric(df[target], errors='coerce').fillna(0)
    t = t.apply(lambda x: np.log(x) if x > 0 else x)
    df[target] = t
    return df


def data_preprocessing_test(df):
    df = df[getLabels()]
    df = feature_preprocessing(df)
    return df


def data_preprocess_train(df):
    train_col = getLabels()
    train_col.append(target)
    df = df[train_col]
    df = feature_preprocessing(df)
    df = target_preprocessing(df)
    return df


# print(train_df.columns)
# print(test_df.columns)

# cat_cols = ['fullVisitorId',
#                'channelGrouping', 'socialEngagementType',
#                'device.isMobile','device.browser', 'device.deviceCategory', 'device.operatingSystem',
#                'geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country', 'geoNetwork.metro', 'geoNetwork.region', 'geoNetwork.subContinent',
#                'totals.bounces', 'totals.newVisits', 'totals.visits',
#                'trafficSource.adContent', 'trafficSource.isTrueDirect', 'trafficSource.medium', 'trafficSource.referralPath', 'trafficSource.source',
#                'date', 'visitStartTime',
#                'visitNumber']

cat_cols = ['fullVisitorId',
               'channelGrouping', 'socialEngagementType',
               'device.browser', 'device.deviceCategory', 'device.operatingSystem',
               'geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country', 'geoNetwork.metro', 'geoNetwork.region', 'geoNetwork.subContinent',
               'totals.bounces', 'totals.newVisits',
               'trafficSource.adContent',  'trafficSource.medium', 'trafficSource.referralPath', 'trafficSource.source']
               # 'date', 'visitStartTime',
               # 'visitNumber' 'device.isMobile','totals.visits' 'trafficSource.isTrueDirect',]


def getBucketSize(size):
    return size*4


def make_feature_cols(train):
    input_labels = []
    for col in cat_cols:
        tc = tf.feature_column.indicator_column(categorical_column_with_hash_bucket(col, getBucketSize(train[col].size)))
        input_labels.append(tc)
    input_labels.append(tf.feature_column.numeric_column('totals.hits'))
    input_labels.append(tf.feature_column.numeric_column('totals.pageviews'))
    input_labels.append(tf.feature_column.numeric_column('totals.visits'))
    return input_labels


def make_input_fn(df, num_epochs, batch_size=128):
    return tf.estimator.inputs.pandas_input_fn(
        x=df[getLabels()],
        y=df[target],
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=True,
        queue_capacity=1000,
        num_threads=1
    )


def make_prediction_input_fn(df, num_epochs):
    return tf.estimator.inputs.pandas_input_fn(
        x=df,
        y=None,
        batch_size=128,
        num_epochs=num_epochs,
        shuffle=True,
        queue_capacity=1000,
        num_threads=1
    )


def add_to_df(predictions):
    p = next(predictions)
    pr = p['predictions']
    return pr[0]

def submission(df):
    submission = df[['fullVisitorId']].copy()
    submission.loc[:, 'PredictedLogRevenue'] = df[target]
    grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
    grouped_test.to_csv('submit.csv', index=False)


def print_rmse(model, name, df):
    metrics = model.evaluate(input_fn=make_input_fn(df, 1))
    print('RMSE on {} dataset = {}'.format(name, np.sqrt(metrics['average_loss'])))
    return metrics['average_loss']


def get_linear_regressor(train):
    myopt = tf.train.FtrlOptimizer(learning_rate=0.1)  # note the learning rate
    estimator = tf.estimator.LinearRegressor(
        model_dir=OUTDIR,
        feature_columns=make_feature_cols(train),
        optimizer=myopt)
    return estimator


def nural_network(train):
    tf.logging.set_verbosity(tf.logging.INFO)
    # model = tf.estimator.DNNRegressor(hidden_units=[32, 16, 8, 4],
    #                                   feature_columns=make_feature_cols(train), model_dir=OUTDIR)

    model = get_linear_regressor(train)

    model.train(input_fn=make_input_fn(train, num_epochs=200, batch_size=50))
    return model


def train_and_evaluate(traindf, testdf):
    estimator = get_linear_regressor(traindf)
    train_spec = tf.estimator.TrainSpec(
        input_fn=make_input_fn(traindf, num_epochs=200, batch_size=50),
        )

    # exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=make_input_fn(testdf, num_epochs=1),
        steps=None,
        start_delay_secs=1,  # start evaluating after N seconds
        throttle_secs=10,  # evaluate every N seconds
        )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def read_csv_chunks(csv_path, nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    chunks = pd.read_csv(csv_path,
                     converters={column: json.loads for column in JSON_COLUMNS},
                     dtype={'fullVisitorId': 'str'},  # Important!!
                     nrows=nrows, chunksize=5000)
    return chunks


OUTDIR = 'k_out_dir'
shutil.rmtree(OUTDIR, ignore_errors=True)  # start fresh each time
def main():
    chunks = read_csv_chunks('data/kagglega/train.csv', nrows=120000)
    i = 1
    avg_loss = 0
    for chunk in chunks:
        try:
            print("chunk = ", i)
            train_df = load_chunked_df(chunk)
            train_df = data_preprocess_train(train_df)
            if i % 2 != 0:
                model = nural_network(train_df)
            else:
                avg_loss = avg_loss + print_rmse(model, 'validation', train_df)
                print('AVG RMSE on {} dataset = {}'.format('total', np.sqrt(avg_loss/(i/4))))
            i = i + 1
        except:
            print("Error in chunk ",i)
            i = i + 1


    test_df = load_df('data/kagglega/test.csv')
    test_df = data_preprocessing_test(test_df)
    test_df[target] = 0

    # model = tf.estimator.DNNRegressor(hidden_units=[32, 16, 8, 4],
    #                                   feature_columns=make_feature_cols(train_df), model_dir=OUTDIR)
    model = tf.estimator.LinearRegressor(
        feature_columns=make_feature_cols(train_df), model_dir = OUTDIR)

    print_rmse(model, 'test', test_df)
    predictions = model.predict(input_fn=make_prediction_input_fn(test_df, 1))
    print(type(predictions))
    test_df[target] = test_df[target].apply(lambda x : add_to_df(predictions))
    submission(test_df)





if __name__ == "__main__":
    main()

