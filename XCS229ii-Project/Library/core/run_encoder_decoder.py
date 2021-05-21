from featuer_generator import *
from utils import *
from trade_strategy import *
from encoder_decoder_model import Encoder_Decoder_Model
from sklearn import metrics
from keras.models import Sequential, load_model

STOCK_SYMBOL = 'GOOG'
START_DATE = '2010-01-01'
END_DATE = '2021-04-30'
ROLLING_X = 3
TRAIN_MODEL = True
SAVE_MODEL_FILE_PATH = "./saved_models/encoder-decoder-21052021-e20-h7-p1.h5"

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
    history_points = 28
    pred_points = 1
    X_train, X_test, y_train, y_test, scaler_y, y_test_date = train_test_split(data=df_input_data,
                                                                               train_test_split_ratio=0.9,
                                                                               features=selected_features,
                                                                               targets=selected_targets,
                                                                               history_points=history_points,
                                                                               pred_points=pred_points)
    # y_train = y_train.reshape(y_train.shape[0], -1)
    # y_test = y_test.reshape(y_test.shape[0], -1)
    if TRAIN_MODEL:
        model = Encoder_Decoder_Model(input_time_steps=history_points, input_dim=len(selected_features),
                                      output_time_steps=pred_points, output_dim=1)
        model.build_model()
        model.train(x=X_train, y=y_train, epochs=20, batch_size=32, save_dir="saved_models")
    else:
        model = load_model(SAVE_MODEL_FILE_PATH)
        if isinstance(model, Sequential):
            print(model.summary())

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
    df_results['pred_err'] = (df_results['pred_price'] - df_results['real_price']) / df_results['real_price']
    df_results['pred_accuracy'] = df_results['pred_err'].abs()

    rmse = int(metrics.mean_squared_error(y_test.reshape(-1), y_pred.reshape(-1), squared=False))
    print("rmse: {}".format(rmse))
    fig_name_1 = "pred_results_rmse_{0}_hist_steps_{1}_pred_steps_{2}".format(rmse, history_points, pred_points)
    plot_time_series_charts(model_name='encoder-decoder', figsize=(20, 8), xlabels=['Date', 'Date'],
                            ylabels=['real_price', 'pred_price'],
                            data=df_results, fig_name=fig_name_1, use_subplots=False,
                            num_xticks=min(len(df_results), 50),
                            num_annotations=min(len(df_results), 10))

    fig_name_2 = "cumulative_returns_hist_steps_{0}_pred_steps_{1}".format(history_points, pred_points)
    plot_time_series_charts(model_name='encoder-decoder', figsize=(20, 8), xlabels=['Date', 'Date'],
                            ylabels=['real_price_cumulative_return',
                                     'pred_price_cumulative_return'],
                            data=df_results, fig_name=fig_name_2, use_subplots=False,
                            num_xticks=min(len(df_results), 50),
                            num_annotations=min(len(df_results), 10))

    fig_name_3 = "pred_accuracy_hist_steps_{0}_pred_steps_{1}".format(history_points, pred_points)
    plot_time_series_charts(model_name='encoder-decoder', figsize=(20, 8), xlabels=['Date'], ylabels=['pred_accuracy'],
                            data=df_results, fig_name=fig_name_3, use_subplots=False,
                            num_xticks=min(len(df_results), 50),
                            num_annotations=min(len(df_results), 10))

    initial_invest = 100000
    buying_percentage_threshold = 0.003
    selling_percentage_threshold = 0.01
    trade_result, passive_trade_result, num_of_stocks, trade_action = buy_sell_trades(
        actual=df_results['real_price'].values.tolist(),
        predicted=df_results['pred_price'].values.tolist(),
        date=df_results['Date'].values.tolist(),
        invest_fund=initial_invest,
        history_points=history_points,
        pred_points=pred_points,
        buying_percentage_threshold=buying_percentage_threshold,
        selling_percentage_threshold=selling_percentage_threshold,
        model_name="encoder_decoder")

    df_results['trade_action'] = trade_action
    fig_name_4 = "trade_action_hist_steps_{0}_pred_steps_{1}".format(history_points, pred_points)
    plot_trade_action(model_name='encoder-decoder', figsize=(20, 8),
                      fig_name=fig_name_4,
                      xlabels=['Date', 'Date'],
                      ylabels=['real_price', 'pred_price'],
                      data=df_results,
                      number_stock=num_of_stocks,
                      initial_invest=initial_invest,
                      passive_trade_result=passive_trade_result,
                      asset=trade_result,
                      rotation=45, num_xticks=50,
                      num_annotations=len(df_results),
                      save_fig=True)
