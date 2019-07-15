import json

from flask import Flask
from flask import request, jsonify

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
    # dfevoo = df[df['product'] == 'extra virgin olive oil (up to 0,8°)']
    # dfevoo = dfevoo[dfevoo['country'] == 'greece']
    # dfevoo=df
    dfevoo['priceStringDate'] = pd.to_datetime(dfevoo['priceStringDate'])
    fevoo = dfevoo.drop(columns=['price_id', 'product', 'priceDate', 'url', 'country', 'dataSource']).sort_values(
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
    train = dataset[0:int(0.8 * (len(dataset))), :]
    valid = dataset[int(0.2 * (len(dataset))):, :]

    # Data Normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Convert Training Data to Right Shape
    lb = 1
    x_train = []
    y_train = []
    # execute a loop that starts from 61st record and stores all the previous 60 records to the x_train list. The 61st record is stored in the y_trainlabels list.
    for i in range(lb, len(train)):
        x_train.append(scaled_data[i - lb:i, 0])
        y_train.append(scaled_data[i, 0])

    # convert both list to the numpy array before training
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # convert our data into the shape accepted by the LSTM 'three-dimensional format'.
    # The first dimension is the number of records or rows in the dataset. The second dimension is the number of time steps which is 60 while the last dimension is the number of indicators
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Training The LSTM
    model = Sequential()  # instantiate the Sequential class.

    # The first parameter to the LSTM layer is the number of neurons that we want in the layer.
    # The second parameter is return_sequences, which is set to true since we will add more layers to the model.
    # The first parameter to the input_shape is the number of time steps while the last parameter is the number of indicators.
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # Dropout layer is added to avoid over-fitting, which is a phenomenon where a machine learning model performs better on the training data compared to the test data
    model.add(Dropout(0.2))

    # add three more LSTM and dropout layers to our model
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100))
    model.add(Dropout(0.2))

    # add a dense layer at the end of the model. The number of neurons in the dense layer will be set to 1 since we want to predict a single value in the output.
    model.add(Dense(1))

    # Model Compilation
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    keras_callbacks = [
        EarlyStopping(monitor='val_mean_absolute_error', patience=20, verbose=0)
    ]
    # Algorithm Training
    epochs = 10
    fitted_model = model.fit(x_train, y_train, epochs=epochs, batch_size=120, verbose=1)
    import pickle

    pickle.dump(fitted_model, open("model_fit.pkl", "wb"), pickle.HIGHEST_PROTOCOL)


    import sqlite3
    conn = sqlite3.connect('Train_results.db')


    c = conn.cursor()
    c.execute('''
       INSERT INTO TrainingResults(Product_Name,Country_Name,Data_Source,Model_fit)
       VALUES (?,?,?,?) ''', (product, country, data_Source, open('model_fit.pkl', 'rb').read()))

    c.execute('''
    SELECT *
    FROM TrainingResults
              ''')
    print(c.fetchall())
    conn.commit()

    return "geiaa"
    # return jsonify(fitted_model.tolist())


@app.route('/prediction', methods=['GET'])
def api_prediction():
    parameters = request.args

    product = parameters.get('product')
    country = parameters.get('country')
    data_Source = parameters.get('data_Source')
    #date = parameters.get('date')
    if not (product or country or data_Source):
        return page_not_found(404)
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
    import sqlite3
    conn = sqlite3.connect('Train_results.db')
    c = conn.cursor()
    results = c.execute(
        'SELECT Model_fit FROM TrainingResults WHERE Product_Name=? and Country_Name=? and Data_Source=?',
        (product, country, data_Source)).fetchall()
    print('to brhka')
    if not results:
        results = api_training()
        print('de to brhka')
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
