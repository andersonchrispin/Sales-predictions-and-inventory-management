import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

def category_prediction(message):
    message['year_month'] = message['date'].apply(lambda x: x[:7])
    message['type'] = message['type'].factorize()[0]
    
    #grouping mean
    message = message.groupby(['year_month', 
                           'store_nbr',]).aggregate({'sales': 'mean', 
                                                     'onpromotion': 'mean', 
                                                     'on_holiday': 'sum',
                                                     'cluster': 'mean',
                                                     'type': 'mean',
                                                     })
    message = message.sort_values(by=['year_month']).reset_index()

    message['cluster'] = message['cluster'].astype(int)
    message.loc[message['onpromotion'].isnull(), 'onpromotion'] = 0
    message['type'] = message['type'].astype(int)
    
    #time order
    message['period'] = message['year_month'].apply(lambda x: x[5:7]).astype(int)

    #saving the entries order
    index = message[['year_month', 'store_nbr']]

    #formatting the features
    dummies = ['store_nbr','on_holiday', 'cluster', 'type']   
    for i in dummies:
        message = pd.concat([message, pd.get_dummies(message[i], prefix=i)], axis=1)
        message = message.drop([i], axis=1)
    message['onpromotion'] = StandardScaler().fit_transform(message['onpromotion'].values.reshape(-1, 1))

    #split data
    y_train = message['sales'][message.year_month < '2017-01']
    x_train = message[message.year_month < '2017-01'].drop(['sales','year_month'], axis=1)
    y_test = message['sales'][message.year_month >= '2017-01']
    x_test = message[message.year_month >= '2017-01'].drop(['sales','year_month'], axis=1)
    
    #training model
    model = RandomForestRegressor(random_state=42, n_estimators=25, max_depth=42)
    model.fit(x_train, y_train)

    y_test_pred = model.predict(x_test)
    test = pd.DataFrame({'y_test': y_test, 'y_test_pred': y_test_pred})
    test[['period','store']] = index
    test = test.reset_index(drop=True)
    
    return test