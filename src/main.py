'''
Created on May 13, 2015

@author: rickerg0
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    
    # turn of data table rendering
    pd.set_option('display.notebook_repr_html', False)
   
    training_set = pd.read_csv('estimate data.txt')
   # print(training_set)
    features = training_set[['tm1','tm2','tm3','tm4','tm5','tm6','tm7','tm8','tm9']]  #.sort_index(by=['est','task'], ascending=[True,True])
    print(features)
    estimations = training_set['est']
    mu = features.mean()
    sigma = features.std()

    features_norm = (features - mu) / sigma
    print(features_norm.head())
   # plt.show()
    #===========================================================================
    # print(foo)
    # my_colors = [(x/10.0, x/70.0, x/10.0) for x in range(len(foo))] 
    # ax = foo.plot(kind='scatter', x=foo.columns.values[0], y=foo.columns.values[10],color='Green', label=foo.columns.values[10],s=100);
    # i=0
    # for t in foo.columns.values[1:10] :
    #     ax= foo.plot(kind='scatter', x='task', y=t, color=my_colors[i], label=t, ax=ax,s=20);
    #     i += 1
    # 
    # plt.show()
    #===========================================================================
    
    
    m = len(features_norm)  # number of data points
    features_norm['x0'] = pd.Series(np.ones(m))
    n = features_norm.columns.size  # number of features
    print(features_norm.head())
    
    alpha = 0.25
    iterations = 150
    m = len(estimations) * 1.0
    
    # 4. Initialize the theta values to zero
    thetas = np.zeros(len(features_norm.columns))
    print(thetas)
    
    # 5. Calculate the theta's by performing Gradient Descent
    foo = np.array(features_norm)
    estimations = np.array(estimations)
    print(estimations)
    cost_history = []

    for i in range(iterations):
    # Calculate the predicted values
        predicted = np.dot(features_norm, thetas)

        # Calculate the theta's for this iteration:
        thetas -= (alpha / m) * np.dot((predicted - estimations), features_norm)
    
    # Calculate cost
        sum_of_square_errors = np.square(predicted - estimations).sum()
        cost = sum_of_square_errors / (2 * m)

        # Append cost to history
        cost_history.append(cost)
    
    print(thetas)
    plt.plot(cost_history[:25])
    plt.show()
    training_set['predictions'] = np.dot(features_norm, thetas)
    training_set['difference'] = training_set['predictions'] - training_set['est']
    print(training_set.head())
    sns.distplot(training_set.est)
    plt.show()
    
    p = sns.lmplot("predictions", "est", data=training_set, size=7)
    p.set_axis_labels("Predicted est", "Observed est")
    plt.show()
    
    p = sns.residplot(training_set.predictions, training_set.est, lowess=True)
    plt.show()
    #===========================================================================