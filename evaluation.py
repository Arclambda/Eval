#Functions for ML in the domain of Insurance
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d



class Evaluation():
    """This function does something cool
    it contains gini(), summaryStats() and lorenz()
    """
    def __init__(self,actual,predicted):
        self.actual = actual
        self.prediction = predicted
        self.giniDF = None
        self.df = pd.DataFrame({'predicted':predicted,'actual':actual})
        



    def gini(self,xPoints=100):
        actual = self.actual
        predicted = self.prediction

        df = self.df

        df = df.sort_values(by='predicted')
        df.reset_index(inplace=True)
        
        df['actualCumulative'] = np.array(df['actual']).cumsum()
        df['percentage'] = df['actualCumulative'] / df['actualCumulative'].max() *100   
        
        
        
        df['percentile'] = np.array(df.index) / np.array(df.index).max() *100
        self.giniDF = df
        # print(df)

        lorenzCurveFunction = interp1d(x = df['percentile'], y = df['percentage'],bounds_error=False,fill_value='extrapolate')
        interpX_axis = np.linspace(0,100,xPoints)
        interY__axis = lorenzCurveFunction(interpX_axis)
    

        return round((np.array(interpX_axis) - np.array(lorenzCurveFunction(interpX_axis))).sum()/xPoints,3)


    def summaryStats(self):

        actual = self.actual
        predicted = self.prediction


        try:
            tweedie = round(100*sklearn.metrics.d2_tweedie_score(actual, predicted, power=1),2)
        except:
            tweedie = 'PowerError'

        try:
            poissionDev = sklearn.metrics.mean_poisson_deviance(actual, predicted),
        except:
            poissionDev = 'PowerError'



        statDict = {
                "Explained variance score 👆" : sklearn.metrics.explained_variance_score(actual, predicted),
                "Max error 👇" : sklearn.metrics.max_error(actual, predicted),
                "Mean absolute error 👇" : sklearn.metrics.mean_absolute_error(actual, predicted),
                "Mean squared error 👇" : sklearn.metrics.mean_squared_error(actual, predicted),
                "Mean squared log error 👇" : sklearn.metrics.mean_squared_log_error(actual, predicted),
                "Median absolute error 👇":sklearn.metrics.median_absolute_error(actual, predicted),
                "R² score [%] 👆" : round(100*sklearn.metrics.r2_score(actual, predicted),2),
                "D² Tweedie score(power=1.9)[%]👆": tweedie,
                "Mean Poisson deviance 👇": poissionDev,
                "Pinball loss (α=0.5) 👇": sklearn.metrics.mean_pinball_loss(actual, predicted, alpha=0.5),
                "Gini Coefficient 👆"   : self.gini
        }

        
        return pd.DataFrame(statDict,[0]).T


    def lorenz(self):
        actual = self.actual
        predicted = self.prediction

        df = pd.DataFrame({'predicted':predicted,'actual':actual})
        df.sort_values(by='predicted',inplace=True)
        df.reset_index(inplace=True)

        df['actualCumulative'] = np.array(df['actual']).cumsum()
        df['percentage'] = df['actualCumulative'] / df['actualCumulative'].max() *100   



        df['percentile'] = np.array(df.index) / np.array(df.index).max() *100
        # print(df)

        lorenzCurveFunction = interp1d(x = df['percentile'], y = df['percentage'],bounds_error=False,fill_value='extrapolate')
        interpX_axis = np.linspace(0,100,100)
        interY__axis = lorenzCurveFunction(interpX_axis)

        lorenzDF = pd.DataFrame({'percentile':interpX_axis,'percentage':interY__axis})

        pxFig = px.line(lorenzDF,x='percentile',y='percentage',template='plotly_dark')
        pxFig.update_layout({'width':900,'height':600})
        pxFig.add_trace(go.Scatter(x=interpX_axis,y=interpX_axis,))

        pxFig.data[0]['showlegend'] = True
        pxFig.data[0]['name'] = 'Catboost ' + '(Gini = ' + str(self.gini) + ')'
        pxFig.data[1]['name'] = 'Perfect Gini'

        return pxFig
