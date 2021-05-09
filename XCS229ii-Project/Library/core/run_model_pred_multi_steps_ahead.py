from featuer_generator import *
from utils import *
from lstm_model import Model
from sklearn import preprocessing
from sklearn import metrics


def train_test_split(data: pd.DataFrame, train_test_split_ratio: float, features: list, targets: list,
                     history_points: int, pred_points: int):
    data = data.dropna(axis=0)

    train_indices = int(data.shape[0] * train_test_split_ratio)
    train_data_x = data.get(features)[:train_indices]
    test_data_x = data.get(features)[train_indices:]
    train_data_y = data.get(targets)[:train_indices]
    test_data_y_date = data.get(['Date'])[train_indices:]
    test_data_y = data.get(targets)[train_indices:]

    train_data_x.reset_index(inplace=True)
    train_data_x = train_data_x.drop(columns=['index'])
    test_data_x.reset_index(inplace=True)
    test_data_x = test_data_x.drop(columns=['index'])
    train_data_y.reset_index(inplace=True)
    train_data_y = train_data_y.drop(columns=['index'])
    test_data_y_date.reset_index(inplace=True)
    test_data_y_date = test_data_y_date.drop(columns=['index'])
    test_data_y.reset_index(inplace=True)
    test_data_y = test_data_y.drop(columns=['index'])

    scaler_x = preprocessing.MinMaxScaler()
    train_data_normalized_x = scaler_x.fit_transform(train_data_x)
    test_data_normalized_x = scaler_x.transform(test_data_x)

    scaler_y = preprocessing.MinMaxScaler()
    train_data_normalized_y = scaler_y.fit_transform(train_data_y)
    test_data_normalized_y = scaler_y.transform(test_data_y)

    data_windows = []
    for i in range(0, train_data_normalized_x.shape[0] - history_points - pred_points, pred_points):
        data_windows.append(train_data_normalized_x[i:i + history_points])
    X_train = np.array(data_windows).astype(float)

    data_windows = []
    for i in range(0, test_data_normalized_x.shape[0] - history_points - pred_points, pred_points):
        data_windows.append(test_data_normalized_x[i:i + history_points])
    X_test = np.array(data_windows).astype(float)

    data_windows = []
    for i in range(history_points, train_data_normalized_y.shape[0] - pred_points, pred_points):
        data_windows.append(train_data_normalized_y[i:i + pred_points])
    y_train = np.array(data_windows).astype(float)

    data_windows = []
    for i in range(history_points, test_data_normalized_y.shape[0] - pred_points, pred_points):
        data_windows.append(test_data_normalized_y[i:i + pred_points])
    y_test = np.array(data_windows).astype(float)

    y_test_date = []
    for i in range(history_points, y_test.shape[0] * pred_points + history_points, 1):
        y_test_date.append(test_data_y_date['Date'].values.tolist()[i])

    assert len(y_test_date) == y_test.shape[0] * pred_points

    return X_train, X_test, y_train, y_test, scaler_y, y_test_date


if __name__ == '__main__':
    df_input_data = get_raw_data(stock_symbol=STOCK_SYMBOL, start_date=START_DATE, end_date=END_DATE, plot_data=True)
    simple_features, df_input_data = derive_simple_features(data=df_input_data, rolling_x=7,
                                                            plot_data=True)
    technical_indicator_features, df_input_data = get_technical_indicator_features(data=df_input_data,
                                                                                   rolling_x=7, plot_data=True)

    selected_features = simple_features + technical_indicator_features
    fig_name_0 = "corr_heat_map"
    plot_corr_heatmap(fig_name=fig_name_0, cols=selected_features, data=df_input_data, save_fig=True)
    selected_targets = ['Adj Close']
    history_points = 7
    pred_points = 4
    X_train, X_test, y_train, y_test, scaler_y, y_test_date = train_test_split(data=df_input_data,
                                                                               train_test_split_ratio=0.9,
                                                                               features=selected_features,
                                                                               targets=selected_targets,
                                                                               history_points=history_points,
                                                                               pred_points=pred_points)
    y_train = y_train.reshape(y_train.shape[0], -1)
    y_test = y_test.reshape(y_test.shape[0], -1)
    model = Model(input_time_steps=history_points, input_dim=len(selected_features), output_dim=pred_points)
    model.build_model()
    model.train(x=X_train, y=y_train, epochs=20, batch_size=32, save_dir="saved_models")
    y_pred = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    df_results = pd.DataFrame()
    df_results['real_price'] = pd.Series(y_test.reshape(-1))
    df_results['pred_price'] = pd.Series(y_pred.reshape(-1))
    df_results['Date'] = pd.Series(y_test_date)
    df_results['real_price_daily_return'] = df_results['real_price'].pct_change().values
    df_results['real_price_cumulative_return'] = (1 + df_results['real_price_daily_return']).cumprod() - 1
    df_results['pred_price_daily_return'] = df_results['pred_price'].pct_change().values
    df_results['pred_price_cumulative_return'] = (1 + df_results['pred_price_daily_return']).cumprod() - 1
    df_results['pred_err'] = (df_results['pred_price'] - df_results['real_price'])/df_results['real_price']
    df_results['pred_accuracy'] =df_results['pred_err'].abs()

    rmse = int(metrics.mean_squared_error(y_test.reshape(-1), y_pred.reshape(-1), squared=False))
    print("rmse: {}".format(rmse))
    fig_name_1 = "pred_results_rmse_{0}_hist_steps_{1}_pred_steps_{2}".format(rmse, history_points, pred_points)
    plot_time_series_charts(figsize=(20, 8), xlabels=['Date', 'Date'], ylabels=['real_price', 'pred_price'],
                            data=df_results, fig_name=fig_name_1, use_subplots=False)

    fig_name_2 = "cumulative_returns_hist_steps_{0}_pred_steps_{1}".format(history_points, pred_points)
    plot_time_series_charts(figsize=(20, 8), xlabels=['Date', 'Date'], ylabels=['real_price_cumulative_return',
                                                                                'pred_price_cumulative_return'],
                            data=df_results, fig_name=fig_name_2, use_subplots=False)

    fig_name_3 = "pred_accuracy_hist_steps_{0}_pred_steps_{1}".format(history_points, pred_points)
    plot_time_series_charts(figsize=(20, 8), xlabels=['Date'], ylabels=['pred_accuracy'],
                            data=df_results, fig_name=fig_name_3, use_subplots=False)
