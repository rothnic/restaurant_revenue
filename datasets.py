__author__ = 'Nick'

import pandas as pd
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')


def add_date_features(data, date_col):

    # convert to datetime, then add feature columns
    dates = pd.to_datetime(date_col)
    data['Year'] = dates.dt.year
    data['Month'] = dates.dt.month
    data['Week'] = dates.dt.weekofyear
    return data


def gen_encoders(data, cols):
    """Create dict of encoders, one for each column in cols"""
    les = {}

    for col in cols:
        le = LabelEncoder()
        le.fit(data[col].values)
        les[col] = le
    return les


def encode(data, encoders):
    for k, v in encoders.iteritems():
        data[k] = v.transform(data[k])

    return data


def get_data():
    # select only features we want and add date info
    train_attr = add_date_features(train.ix[:, 2:-1], train['Open Date'])
    target = train.ix[:, -1]
    test_attr = add_date_features(test.ix[:, 2:], test['Open Date'])

    all_data = pd.concat([train_attr, test_attr])
    enc_cols = ['City', 'City Group', 'Type']
    encoders = gen_encoders(all_data, enc_cols)

    train_attr = encode(train_attr, encoders)
    test_attr = encode(test_attr, encoders)

    return train_attr, test_attr, target



