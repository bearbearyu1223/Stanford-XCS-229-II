import os
import json
import matplotlib.pyplot as plt
from data_preprocessor import DataLoader
from model import Model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label="True Data", color="blue")
    # Pad the list of predictions to shift it in the graph to it's correct start
    predicted_data_all = []
    for i, data in enumerate(predicted_data):
        if i == 0:
            padding = [None for p in range(prediction_len)]
            predicted_data_all = predicted_data_all + padding
        predicted_data_all = predicted_data_all + data
        # padding = [None for p in range(i * prediction_len)]
        # plt.plot(padding + data, label="Prediction", linestyle="--")
        # plt.legend()
    plt.plot(predicted_data_all, label="Prediction", linestyle="--")
    plt.show()


def main():
    configs = json.load(open('../config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('../data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    model = Model()
    model.build_model(configs)
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalised=configs['data']['normalise']
    )

    model.train(
        x,
        y,
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        save_dir=configs['model']['save_dir']
    )

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalised=configs['data']['normalise']
    )

    predictions = model.predict_multiple_steps_head(data=x_test,
                                                    window_size=configs['data']['sequence_length'],
                                                    prediction_len=configs['data']['sequence_length'])

    plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])


if __name__ == '__main__':
    main()
