import json
import pickle
import pandas as pd
import numpy as np
from flask import Flask
from flask import request, jsonify

from api.utils.LSTMModel import LSTMModel

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/train', methods=['GET'])
def api_training():
    parameters = request.args

    product = parameters.get('product')
    country = parameters.get('country')
    data_Source = parameters.get('data_Source')

    import pandas as pd
    import numpy as np

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
    scaler = MinMaxScaler(feature_range=(0, 1))

    sns.set_style('darkgrid')
    df = pd.read_csv("food_dataset.csv")
    dfevoo = df
    if product:
        dfevoo = df[df['product'] == product]
    if country:
        dfevoo = dfevoo[dfevoo['country'] == country]
    if data_Source:
        dfevoo = dfevoo[dfevoo['dataSource'] == data_Source]

    if not (product or country or data_Source):
        return page_not_found(404)

    dfevoo['priceStringDate'] = pd.to_datetime(dfevoo['priceStringDate'])
    dfevoo = dfevoo.drop(columns=['price_id', 'product', 'priceDate', 'url', 'country', 'dataSource']).sort_values(
        by='priceStringDate')
    dfevoo = pd.DataFrame(dfevoo)
    dfevoo = dfevoo.groupby('priceStringDate').mean().reset_index()
    Data = dfevoo

    # Long Short Term Memory (LSTM)

    # setting index
    Data.index = Data.priceStringDate
    Data.drop('priceStringDate', axis=1, inplace=True)

    # creating train and test sets
    dataset = Data.values
    lb = 80

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(dataset)

    test = dataset[len(dataset) - lb:, :]
    dataset = dataset[:len(dataset) - lb, :]
    train = dataset[:len(dataset) - lb, :]
    # valid = dataset[int(0.2 * (len(dataset))):, :]
    scaled_data = scaler.transform(train)
    # Convert Training Data to Right Shape
    x_train = []
    y_train = []
    for i in range(lb, len(train)):
        x_train.append(scaled_data[i - lb:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = LSTMModel(x_train=x_train)

    keras_callbacks = [
        EarlyStopping(monitor='loss', patience=20, verbose=0)
    ]
    config = {}
    config['epochs'] = 1000
    config['batch_size'] = 32
    config['verbose'] = 1
    config['callbacks'] = keras_callbacks

    model.fit(features=x_train, labels=y_train, config=config)

    predictions = []
    for preds in range(0, lb + 1):

        valid = dataset[len(dataset) - lb:]
        inputs = valid

        # Scale our test data
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        # final test input
        X_test = []
        for i in range(lb, inputs.shape[0] + 1):
            X_test.append(inputs[i - lb:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        closing_price = scaler.inverse_transform(model.predict(data=X_test))

        predictions.append(closing_price.flatten()[0])

        dataset = np.append(dataset, np.array([closing_price.flatten()[0]]))

    plt.plot(predictions, color='green', label='Predicted')
    plt.plot(test, color='blue', label='Actual')
    plt.title('Price Prediction  ')
    plt.xlabel('date')
    plt.ylabel('price')
    plt.legend()
    plt.savefig('plots/pred_1_%s.png' % str(lb), bbox_inches='tight')
    plt.show()

    complete_pred = []
    for val in train:
        complete_pred.append(val)
    for pred in predictions:
        complete_pred.append(pred)

    actual = []
    for val in train:
        actual.append(val)
    for val in test:
        actual.append(val)

    plt.plot(complete_pred, color='green', label='Predicted')
    plt.plot(actual, color='blue', label='Actual')
    plt.title('Price Prediction  ')
    plt.xlabel('date')
    plt.ylabel('price')
    plt.legend()
    plt.savefig('plots/pred_2_%s.png' % str(lb), bbox_inches='tight')
    plt.show()

    dictionary = {'predictions': predictions}
    return json.dumps(str(dictionary))


# import pickle
#
# pickle.dump(fitted_model, open("model_fit.pkl", "wb"), pickle.HIGHEST_PROTOCOL)
#
# import sqlite3
#
# conn = sqlite3.connect('Train_results.db')
#
# c = conn.cursor()
# c.execute('''
#        INSERT INTO TrainingResults(Product_Name,Country_Name,Data_Source,Model_fit)
#        VALUES (?,?,?,?) ''', (product, country, data_Source, open('model_fit.pkl', 'rb').read()))
#
# c.execute('''
#     SELECT *
#     FROM TrainingResults
#               ''')
# print(c.fetchall())
# conn.commit()
#
# return "geiaa"
# return jsonify(fitted_model.tolist())


@app.route('/prediction', methods=['GET'])
def api_prediction():
    # parameters = request.args
    #
    # product = parameters.get('product')
    # country = parameters.get('country')
    # data_Source = parameters.get('data_Source')
    # # date = parameters.get('date')
    # if not (product or country or data_Source):
    #     return page_not_found(404)

    #
    # from keras.models import Sequential
    # from keras.layers import Dense
    # from keras.layers import Dropout
    # from keras.layers import LSTM
    # from sklearn.preprocessing import MinMaxScaler
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
    # scaler = MinMaxScaler(feature_range=(0, 1))
    #
    # sns.set_style('darkgrid')
    # import sqlite3
    # conn = sqlite3.connect('Train_results.db')
    # c = conn.cursor()
    # results = c.execute(
    #     'SELECT Model_fit FROM TrainingResults WHERE Product_Name=? and Country_Name=? and Data_Source=?',
    #     (product, country, data_Source)).fetchall()
    # print('to brhka')
    # if not results:
    #     results = api_training()
    #     print('de to brhka')

    model = pickle.load(open('model_fit.pkl', 'rb'))
    dfevoo = pd.read_csv("food_dataset.csv")
    dfevoo['priceStringDate'] = pd.to_datetime(dfevoo['priceStringDate'])
    fevoo = dfevoo.drop(columns=['price_id', 'product', 'priceDate', 'url', 'country', 'dataSource']).sort_values(
        by='priceStringDate')
    dfevoo = pd.DataFrame(dfevoo)
    dfevoo = dfevoo.groupby('priceStringDate').mean().reset_index()
    Data = dfevoo
    # Prepare our test inputs
    dataset = Data.values
    lb = 1
    valid = dataset[int(0.2 * (len(dataset))):, :]
    inputs = Data[len(Data) - len(valid) - lb:].values

    # Scale our test data
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    # final test input
    X_test = []
    for i in range(lb, inputs.shape[0]):
        X_test.append(inputs[i - lb:i, 0])

    closing_price = model.predict(X_test)
    return jsonify(closing_price.flatten())
    #
    #     dfevoo = pd.read_csv("food_dataset.csv")
    #     dfevoo['priceStringDate'] = pd.to_datetime(dfevoo['priceStringDate'])
    #     fevoo = dfevoo.drop(columns=['price_id', 'product', 'priceDate', 'url', 'country', 'dataSource']).sort_values(
    #         by='priceStringDate')
    #     dfevoo = pd.DataFrame(dfevoo)
    #     dfevoo = dfevoo.groupby('priceStringDate').mean().reset_index()
    #     Data = dfevoo
    #     # Prepare our test inputs
    #     dataset = Data.values
    #     lb = 1
    #     valid = dataset[int(0.2 * (len(dataset))):, :]
    #     inputs = Data[len(Data) - len(valid) - lb:].values
    #
    #     # Scale our test data
    #     inputs = inputs.reshape(-1, 1)
    #     inputs = scaler.transform(inputs)
    #
    #     # final test input
    #     X_test = []
    #     for i in range(lb, inputs.shape[0]):
    #         X_test.append(inputs[i - lb:i, 0])
    #
    #     # Convert our data into the three-dimensional format which can be used as input to the LSTM.
    #     X_test = np.array(X_test)
    #     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #
    #     # Making Predictions
    #     closing_price = model.predict(X_test)
    #     # reverse the scaled prediction back to their actual values.
    #     # Use the ìnverse_transform method of the scaler object we created during training
    #     closing_price = scaler.inverse_transform(closing_price)
    #
    #
    # # return jsonify(closing_price.tolist())
    #     return jsonify(closing_price)
    return "j"


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404


if __name__ == '__main__':
    app.run()
