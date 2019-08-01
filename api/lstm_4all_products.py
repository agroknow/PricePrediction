import requests
import json
import pandas as pd
from api.utils.LSTMModel import LSTMModel
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import os
import numpy as np

scaler = MinMaxScaler(feature_range=(0, 1))

# take json data from API
response = requests.get("http://148.251.22.254:8080/price-api-1.0/price/findAll")
data = response.text
parsed = json.loads(data)
# print(json.dumps(parsed, indent=4))

# dataframe
df = pd.DataFrame(parsed)
prouped_ordered = df.groupby(['product']).size().reset_index(name='counts').sort_values('counts')
print(prouped_ordered)
# \test=prouped_ordered[800 :]
prouped_ordered = prouped_ordered[prouped_ordered['counts'] > 1000]
# quit(0)
for product in prouped_ordered['product']:
    print(product)

    path = os.getcwd()
    path=path+'/'+product+'/'
    os.mkdir(path)
    print("The current working directory is %s" % path)
    # quit(0)
    dfevoo = df[df['product'] == product]
    print(dfevoo)
    dfevoo = pd.DataFrame(dfevoo)

    dfevoo['priceStringDate'] = pd.to_datetime(dfevoo['priceStringDate'])
    dfevoo = dfevoo.drop(columns=['price_id', 'product', 'priceDate', 'url', 'country', 'dataSource']).sort_values(
        by='priceStringDate')
    dfevoo = pd.DataFrame(dfevoo)
    dfevoo = dfevoo.groupby('priceStringDate').mean().reset_index()
    Data = dfevoo
    print(Data)
    Data.index = Data.priceStringDate
    Data.drop('priceStringDate', axis=1, inplace=True)

    # lb = 80
    rms = [[]]

    for lb in range(55, 66, 1):
        # creating train and test sets
        dataset = Data.values
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
        plt.savefig(path+'pred_1_%s.png' % str(lb), bbox_inches='tight')
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
        plt.savefig(path+'pred_2_%s.png' % str(lb), bbox_inches='tight')
        plt.show()
        rmse = np.sqrt(np.mean(np.power((test - closing_price), 2)))
        rms.append([rmse, lb])

        print(rms)
    # quit(0)
