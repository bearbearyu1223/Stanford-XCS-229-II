def buy_sell_trades(actual: list, predicted: list, history_points: int, pred_points: int, date: list,
                    invest_fund: float, buying_percentage_threshold=0.0015,
                    selling_percentage_threshold=0.0015, initial_number_of_stocks=0, save_log=True):
    y_pct_change = [a1 / a2 - 1.0 for a1, a2 in zip(predicted[1:], predicted)]

    number_of_stocks = initial_number_of_stocks

    trade_result = invest_fund

    passive_trade_result = invest_fund - int(invest_fund / actual[0]) * actual[0] + actual[
        len(actual) - 1] * (initial_number_of_stocks + int(invest_fund / actual[0]))

    trade_action = []  # -1 for selling, 1 for buying, 0 for holding
    if save_log:
        f = open("./logs/trading_history_hist_{}_pred_{}.txt".format(history_points, pred_points), 'w+')
    for i in range(len(actual) - 1):
        if i + 1 < len(y_pct_change) and y_pct_change[i + 1] > buying_percentage_threshold:
            k = 0
            while trade_result - actual[i] > 0:
                trade_result = trade_result - actual[i]
                number_of_stocks = number_of_stocks + 1
                k = k + 1
            if k > 0:
                trade_action.append(1)
            else:
                trade_action.append(0)
            if save_log:
                temp = trade_result + number_of_stocks * actual[i]
                f.write(
                    'Date : {: <12} Activity: {: <8} Stocks Trade: {: <8} Total Stocks: {: <8} Price: {:<8} Total '
                    'Assets: ${:<8} \n'.format(
                        date[i], "BUY" if k > 0 else "HOLD", k,
                        number_of_stocks, int(actual[i]), int(temp)))
        elif i + 1 < len(y_pct_change) and y_pct_change[i + 1] < -selling_percentage_threshold:
            k = 0
            while number_of_stocks > 0:
                trade_result = trade_result + actual[i]
                number_of_stocks = number_of_stocks - 1
                k = k + 1
            if k > 0:
                trade_action.append(-1)
            else:
                trade_action.append(0)
            if save_log:
                temp = trade_result + number_of_stocks * actual[i]
                f.write(
                    'Date : {: <12} Activity: {: <8} Stocks Trade: {: <8} Total Stocks: {: <8} Price: {:<8} Total '
                    'Assets: ${:<8} \n'.format(
                        date[i], "SELL" if k > 0 else "HOLD", k,
                        number_of_stocks, int(actual[i]), int(temp)))
        else:
            k = 0
            trade_action.append(0)
            if save_log:
                temp = trade_result + number_of_stocks * actual[i]
                f.write(
                    'Date : {: <12} Activity: {: <8} Stocks Trade: {: <8} Total Stocks: {: <8} Price: {:<8} Total '
                    'Assets: ${:<8} \n'.format(
                        date[i], "HOLD", k,
                        number_of_stocks, int(actual[i]), int(temp)))
    trade_action.append(0)
    if save_log:
        f.close()
    trade_result = trade_result + number_of_stocks * actual[len(actual) - 1]

    return trade_result, passive_trade_result, number_of_stocks, trade_action
