#Functions for ML in the domain of Insurance
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from plotly.subplots import make_subplots
import plotly.graph_objs as go



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



class Graphing():
    """This function does something cool
    it contains gini(), summaryStats() and lorenz()
    """
    def __init__(self,actual,glmPred,catBoostPred,testSet):
        # self.actual = actual
        # self.prediction = predicted
        if not ((len(glmPred) == len(catBoostPred)) & (len(catBoostPred) == len(testSet))):
            raise ValueError("Arrays and/or df not same length!")
        self.glmPred = glmPred
        self.catBoostPred = catBoostPred
        self.testSet = testSet
        self.actual = actual

        difference = catBoostPred - glmPred
        dfs = pd.DataFrame({'GLMPred':glmPred,'catboostPred':catBoostPred,'difference':difference,'percentageChange':(difference/glmPred)*100})
        self.df = pd.concat([testSet,dfs],axis=1)

        df = self.df
    

    def genderGraph(self,df):

        
    
        rateDF = df[['DRI_Gender','GLMPred','catboostPred']].groupby('DRI_Gender').mean()
        countDF = pd.DataFrame({'count':df['DRI_Gender'].value_counts(),'percentTotal': df['DRI_Gender'].value_counts(normalize=True) * 100})

        rateDF['difference'] = list(rateDF['catboostPred'] - rateDF['GLMPred'])
        rateDF['rateChange'] = list((rateDF['difference'] / rateDF['GLMPred']))

        x0 = countDF.index
        
        hoverTemplateBar = 'Count: %{y}<br>%{customdata:.2f}%'
        graphOne = go.Bar(x=x0,y=countDF['count'][x0],customdata=countDF['percentTotal'][x0],marker={'color':'skyblue'},name='Exposure',hovertemplate=hoverTemplateBar)
        graphTwo = go.Scatter(x=x0,y=rateDF['rateChange'][x0],line={"width":4,"color":"rgb(220, 0, 110)"},name='% Rate Change',hovertemplate='Change: %{y}')
        return graphOne, graphTwo

    def sumInsuredGraph(self,df):
        #Create bin ranges so as to allocate each row to a bin
        bins = list(range(0,1550000,50000))
        binStrings = [str(bins[x])+'-'+str(bins[x+1]) for x in range(len(bins)-1)]
        # pd.cut(df['VEH_SumInsured'],bins=bins,labels=binStrings)

        df['sumInsuredBin'] = pd.cut(df['VEH_SumInsured'],bins=bins,labels=binStrings)


        rateDF = df[['VEH_SumInsured','sumInsuredBin','GLMPred','catboostPred']].groupby('sumInsuredBin').mean()
        countDF = pd.DataFrame({'count':df['sumInsuredBin'].value_counts(),'percentTotal': df['sumInsuredBin'].value_counts(normalize=True) * 100})

        rateDF['difference'] = list(rateDF['catboostPred'] - rateDF['GLMPred'])
        rateDF['rateChange'] = list((rateDF['difference'] / rateDF['GLMPred']))

        x0 = [binStrings[:-10]][0]

        #Customising the hover template
        hoverTemplateBar = 'Count: %{y}<br>%{customdata:.2f}%'
        hoverTemplateBar2 = 'Change: %{y}<br>Category: %{customdata}'

        #Instantiating the graphs with parameters
        graphOne = go.Bar(x=x0,y=countDF['count'][x0],marker={'color':'skyblue'},customdata=countDF['percentTotal'][x0],name='Exposure',hovertemplate=hoverTemplateBar)
        graphTwo = go.Scatter(x=x0,y=rateDF['rateChange'][x0],customdata=list(x0),line={"width":4,"color":"rgb(220, 0, 110)"},name='% Rate Change',hovertemplate=hoverTemplateBar2)
        return graphOne, graphTwo


    def licenseCodeGraph(self,df):

        feature = 'DRI_LicenseCode'   


        rateDF = df[[feature,'GLMPred','catboostPred']].groupby(feature).mean()
        countDF = pd.DataFrame({'count':df[feature].value_counts(),'percentTotal': df[feature].value_counts(normalize=True) * 100})

        rateDF['difference'] = list(rateDF['catboostPred'] - rateDF['GLMPred'])
        rateDF['rateChange'] = list((rateDF['difference'] / rateDF['GLMPred']))

        x0 = countDF.index

        #Customising the hover template
        hoverTemplateBar = 'Count: %{y}<br>%{customdata:.2f}%'
        hoverTemplateBar2 = 'Change: %{y}<br>Category: %{customdata}'
        
        #Instantiating the graphs with parameters
        graphOne = go.Bar(x=x0,y=countDF['count'][x0],marker={'color':'skyblue'},customdata=countDF['percentTotal'],name='Exposure',hovertemplate=hoverTemplateBar)
        graphTwo = go.Scatter(x=x0,y=rateDF['rateChange'][x0].values,customdata=list(x0),line={"width":4,"color":"rgb(220, 0, 110)"},name='% Rate Change',hovertemplate=hoverTemplateBar2)
        return graphOne,graphTwo


    def itcScoreGraph(self,df):
        feature = 'POL_ITCScore_Org'

        #Create bin ranges so as to allocate each row to a bin
        bins = list(range(0,df[feature].max()+50,50))
        binStrings = [str(bins[x])+'-'+str(bins[x+1]) for x in range(len(bins)-1)]
        bins2 = pd.cut(df[feature],bins=bins,labels=binStrings,include_lowest=True)
        

        df['ITCBin'] = pd.cut(df[feature],bins=bins,labels=binStrings,include_lowest=True)


        rateDF = df[[feature,'ITCBin','GLMPred','catboostPred']].groupby('ITCBin').mean()
        countDF = pd.DataFrame({'count':df['ITCBin'].value_counts(),'percentTotal': df['ITCBin'].value_counts(normalize=True) * 100})

        rateDF['difference'] = list(rateDF['catboostPred'] - rateDF['GLMPred'])
        rateDF['rateChange'] = list((rateDF['difference'] / rateDF['GLMPred']))

        x0 = binStrings

        #Customising the hover template
        hoverTemplateBar = 'Count: %{y}<br>%{customdata:.2f}%'
        hoverTemplateBar2 = 'Change: %{y}<br>Category: %{customdata}'
        
        #Instantiating the graphs with parameters
        graphOne = go.Bar(x=x0,y=countDF['count'][x0],marker={'color':'skyblue'},customdata=countDF['percentTotal'][x0],name='Exposure',hovertemplate=hoverTemplateBar)
        graphTwo = go.Scatter(x=x0,y=rateDF['rateChange'][x0],customdata=list(x0),line={"width":4,"color":"rgb(220, 0, 110)"},name='% Rate Change',hovertemplate=hoverTemplateBar2)

        return graphOne, graphTwo

    def driverAgeGraph(self,df):

        feature = 'DRI_Age'
        binName = 'ageBin'

        #Create bin ranges so as to allocate each row to a bin
        bins = list(range(18,101,5))
        binStrings = [str(bins[x])+'-'+str(bins[x+1]) for x in range(len(bins)-1)]
        bins2 = pd.cut(df[feature],bins=bins,labels=binStrings,include_lowest=True)

        

        df[binName] = pd.cut(df[feature],bins=bins,labels=binStrings,include_lowest=True)


        rateDF = df[[feature,binName,'GLMPred','catboostPred']].groupby(binName).mean()
        countDF = pd.DataFrame({'count':df[binName].value_counts(),'percentTotal': df[binName].value_counts(normalize=True) * 100})

        rateDF['difference'] = list(rateDF['catboostPred'] - rateDF['GLMPred'])
        rateDF['rateChange'] = list((rateDF['difference'] / rateDF['GLMPred']))

        x0 = binStrings

        #Customising the hover template
        hoverTemplateBar = 'Count: %{y}<br>%{customdata:.2f}%'
        hoverTemplateBar2 = 'Change: %{y}<br>Category: %{customdata}'
        
        #Instantiating the graphs with parameters
        graphOne = go.Bar(x=x0,y=countDF['count'][x0],marker={'color':'skyblue'},customdata=countDF['percentTotal'][x0],name='Exposure',hovertemplate=hoverTemplateBar)
        graphTwo = go.Scatter(x=x0,y=rateDF['rateChange'][x0],customdata=list(x0),line={"width":4,"color":"rgb(220, 0, 110)"},name='% Rate Change',hovertemplate=hoverTemplateBar2)
        return graphOne, graphTwo

    def excessGraph(self,df):
        feature = 'VEH_Excess'
        binName = 'excessBin'

        #Create bin ranges so as to allocate each row to a bin
        bins = list(range(0,42000,2000))
        binStrings = [str(bins[x])+'-'+str(bins[x+1]) for x in range(len(bins)-1)]
        bins2 = pd.cut(df[feature],bins=bins,labels=binStrings,include_lowest=True)

        

        df[binName] = pd.cut(df[feature],bins=bins,labels=binStrings,include_lowest=True)


        rateDF = df[[feature,binName,'GLMPred','catboostPred']].groupby(binName).mean()
        countDF = pd.DataFrame({'count':df[binName].value_counts(),'percentTotal': df[binName].value_counts(normalize=True) * 100})

        rateDF['difference'] = list(rateDF['catboostPred'] - rateDF['GLMPred'])
        rateDF['rateChange'] = list((rateDF['difference'] / rateDF['GLMPred']))

        x0 = binStrings

        #Customising the hover template
        hoverTemplateBar = 'Count: %{y}<br>%{customdata:.2f}%'
        hoverTemplateBar2 = 'Change: %{y}<br>Category: %{customdata}'
        
        #Instantiating the graphs with parameters
        graphOne = go.Bar(x=x0,y=countDF['count'][x0],marker={'color':'skyblue'},customdata=countDF['percentTotal'][x0],name='Exposure',hovertemplate=hoverTemplateBar)
        graphTwo = go.Scatter(x=x0,y=rateDF['rateChange'][x0],customdata=list(x0),line={"width":4,"color":"rgb(220, 0, 110)"},name='% Rate Change',hovertemplate=hoverTemplateBar2)
        return graphOne, graphTwo

    def licenseYearsGraph(self,df):
        feature = 'DRI_LicenseYears_V'
        binName = 'bin'

        #Create bin ranges so as to allocate each row to a bin
        bins = list(range(18,85,5))
        binStrings = [str(bins[x])+'-'+str(bins[x+1]) for x in range(len(bins)-1)]
        bins2 = pd.cut(df[feature],bins=bins,labels=binStrings,include_lowest=True)

        

        df[binName] = pd.cut(df[feature],bins=bins,labels=binStrings,include_lowest=True)


        rateDF = df[[feature,binName,'GLMPred','catboostPred']].groupby(binName).mean()
        countDF = pd.DataFrame({'count':df[binName].value_counts(),'percentTotal': df[binName].value_counts(normalize=True) * 100})

        rateDF['difference'] = list(rateDF['catboostPred'] - rateDF['GLMPred'])
        rateDF['rateChange'] = list((rateDF['difference'] / rateDF['GLMPred']))

        x0 = binStrings

        #Customising the hover template
        hoverTemplateBar = 'Count: %{y}<br>%{customdata:.2f}%'
        hoverTemplateBar2 = 'Change: %{y}<br>Category: %{customdata}'
        
        #Instantiating the graphs with parameters
        graphOne = go.Bar(x=x0,y=countDF['count'][x0],marker={'color':'skyblue'},customdata=countDF['percentTotal'][x0],name='Exposure',hovertemplate=hoverTemplateBar)
        graphTwo = go.Scatter(x=x0,y=rateDF['rateChange'][x0],customdata=list(x0),line={"width":4,"color":"rgb(220, 0, 110)"},name='% Rate Change',hovertemplate=hoverTemplateBar2)
        return graphOne, graphTwo

    def carAgeGraph(self,df):
        feature = 'VEH_CarAge'
        binName = 'bin'

        #Create bin ranges so as to allocate each row to a bin
        bins = list(range(0,27,1))
        binStrings = [str(bins[x])+'-'+str(bins[x+1]) for x in range(len(bins)-1)]
        bins2 = pd.cut(df[feature],bins=bins,labels=binStrings,include_lowest=True)

        

        df[binName] = pd.cut(df[feature],bins=bins,labels=binStrings,include_lowest=True)


        rateDF = df[[feature,binName,'GLMPred','catboostPred']].groupby(binName).mean()
        countDF = pd.DataFrame({'count':df[binName].value_counts(),'percentTotal': df[binName].value_counts(normalize=True) * 100})

        rateDF['difference'] = list(rateDF['catboostPred'] - rateDF['GLMPred'])
        rateDF['rateChange'] = list((rateDF['difference'] / rateDF['GLMPred']))

        x0 = binStrings[:None]
        
        #Customising the hover template
        hoverTemplateBar = 'Count: %{y}<br>%{customdata:.2f}%'
        hoverTemplateBar2 = 'Change: %{y}<br>Category: %{customdata}'
        


        myY = list(countDF['count'][x0])
        for i in range(len(myY)):
            myY[i] = float(myY[i])
            
        #Instantiating the graphs with parameters
        graphOne = go.Bar({'x':{'type' : 'category'}},x=x0,y=countDF['count'][x0],marker={'color':'skyblue'},customdata=countDF['percentTotal'][x0],name='Exposure',hovertemplate=hoverTemplateBar)
            
        graphTwo = go.Scatter({'x':{'type' : 'category'}},x=x0,y=rateDF['rateChange'][x0],customdata=list(x0),line={"width":4,"color":"rgb(220, 0, 110)"},name='% Rate Change',hovertemplate=hoverTemplateBar2)

        return graphOne, graphTwo

    def compDurationGraph(self,df):

        feature = 'DRI_CompInsuranceDuration'
        binName = 'bin'


        rateDF = df[[feature,binName,'GLMPred','catboostPred']].groupby(feature).mean()
        countDF = pd.DataFrame({'count':df[feature].value_counts(),'percentTotal': df[feature].value_counts(normalize=True) * 100})

        rateDF['difference'] = list(rateDF['catboostPred'] - rateDF['GLMPred'])
        rateDF['rateChange'] = list((rateDF['difference'] / rateDF['GLMPred']))

        x0 = sorted(list(df['DRI_CompInsuranceDuration'].unique()))

        hoverTemplateBar = 'Count: %{y}<br>%{customdata:.2f}%'
        hoverTemplateBar2 = 'Change: %{y}<br>Category: %{customdata}'
        
        graphOne = go.Bar(x=x0,y=countDF['count'][x0],marker={'color':'skyblue'},customdata=countDF['percentTotal'][x0],name='Exposure',hovertemplate=hoverTemplateBar)
        graphTwo = go.Scatter(x=x0,y=rateDF['rateChange'][x0],customdata=list(x0),line={"width":4,"color":"rgb(220, 0, 110)"},name='% Rate Change',hovertemplate=hoverTemplateBar2)

        return graphOne, graphTwo

    def cap_difference(self,cap=1):
        
        diff = self.catBoostPred - self.glmPred
        diff = np.where(diff > cap * self.glmPred, cap * self.glmPred, diff)
        diff = np.where(diff < -cap * self.glmPred, -cap * self.glmPred, diff)
        return self.glmPred + diff



    def capGraphs(self):
        caps = np.arange(0, 1.025, 0.025)
        ginis = []

        

        for i in caps:
            capArray = self.cap_difference(i)
            ev = Evaluation(self.actual,capArray)
            gini = ev.gini()
            ginis.append(gini)

        fig = go.Scatter(x=caps,y=ginis,name='Exposure',line={"width":4,"color":"rgb(220, 0, 110)"})
        # fig = go.Figure(giniLine)

        # gridLineColour = "rgb(50,50,50)"

        # fig.update_layout({"height":600,
        #                         "width":1000,
        #                         "plot_bgcolor":'rgb(28,28,28)',
        #                         "paper_bgcolor": 'rgb(28,28,28)',
        #                         "yaxis":{'gridcolor':gridLineColour,'showline':True,'tickfont_size':13,'title_text':'Gini'},
        #                         "yaxis2":{'showgrid':False,'showline':True,'tickformat':'.2%','tickfont_size':13,'title_text':'Cap'},
        #                         "xaxis":{'showgrid':True,'showline':True,'title_text':'Cap','dtick' : 0.05},
        #                         "title":{'text':'Gini Change by Cap','x':0.5,"font_size":22}
        #                         })
        return fig

    def rateChangeGraph(self):
        fig = go.Histogram(x=self.df['percentageChange'],
                            marker={'color':'skyblue'},
                            xbins=dict(end = 60,start = -60,size = 0.5)
                            )
        # fig.update_layout(title_text='Rate Change Distribution',
        #                 title_x = 0.5,
        #                 xaxis_title_text = 'Exposure',
        #                 yaxis_title_text = 'Rate Change',
        #                 template = 'plotly_dark',
        #                 bargap = 0.3,
        #                 height = 600,
        #                 width = 1000
        #                 )
        # fig.update_traces(xbins=dict(end = 60,start = -60,size = 0.5),
        #                 marker_cmax = 100,
        #                 marker_cmid = 0,
        #                 marker_cmin = -100,
        #                 marker_color = 'lightskyblue'
        #                 )
        return fig


    def graphs(self):

        df = self.df
        
        genderGraph = self.genderGraph(df)
        sumInsuredGraph = self.sumInsuredGraph(df)
        licenseCodeGraph = self.licenseCodeGraph(df)
        itcScoreGraph = self.itcScoreGraph(df)
        driveAgeGraph = self.driverAgeGraph(df)
        excessGraph = self.excessGraph(df)
        licenseYears = self.licenseYearsGraph(df)
        carAge = self.carAgeGraph(df)
        compDurationGraph = self.compDurationGraph(df)
        rateChangeGraph = self.rateChangeGraph()
        capGraph = self.capGraphs()

        subplotTitles = ['DRI_Gender','VEH_SumInsured','DRI_LicenseCode','POL_ITCScore_Org','DRI_Age','VEH_Excess','DRI_LicenseYears_V','VEH_CarAge','DRI_CompInsuranceDuration','Rate Change Distribution','Gini by Cap']

        gridLineColour = "rgb(50,50,50)"
        fig2 = make_subplots(specs=[[{"secondary_y": True},{"secondary_y": True}],
                                    [{"secondary_y": True},{"secondary_y": True}],
                                    [{"secondary_y": True},{"secondary_y": True}],
                                    [{"secondary_y": True},{"secondary_y": True}],
                                    [{"secondary_y": True},{"secondary_y": True}],
                                    [{"secondary_y": True},{"secondary_y": True}]],
                                    rows=6,cols=2,subplot_titles=subplotTitles)


        
        fig2.add_trace(genderGraph[0],row = 1,col=1)
        fig2.add_trace(genderGraph[1],row = 1,col=1,secondary_y=True)

        fig2.add_trace(sumInsuredGraph[0],row = 1,col=2)
        fig2.add_trace(sumInsuredGraph[1],row = 1,col=2,secondary_y=True)

        fig2.add_trace(licenseCodeGraph[0],row = 2,col=1)
        fig2.add_trace(licenseCodeGraph[1],row = 2,col=1,secondary_y=True)

        fig2.add_trace(itcScoreGraph[0],row = 2,col=2)
        fig2.add_trace(itcScoreGraph[1],row = 2,col=2,secondary_y=True)

        fig2.add_trace(driveAgeGraph[0],row = 3,col=1)
        fig2.add_trace(driveAgeGraph[1],row = 3,col=1,secondary_y=True)

        fig2.add_trace(excessGraph[0],row = 3,col=2)
        fig2.add_trace(excessGraph[1],row = 3,col=2,secondary_y=True)

        fig2.add_trace(licenseYears[0],row = 4,col=1)
        fig2.add_trace(licenseYears[1],row = 4,col=1,secondary_y=True)

        fig2.add_trace(carAge[0],row = 4,col=2)
        fig2.add_trace(carAge[1],row = 4,col=2,secondary_y=True)
        fig2.update_xaxes(row = 4,col=2,type='category')

        fig2.add_trace(compDurationGraph[0],row = 5,col=1)
        fig2.add_trace(compDurationGraph[1],row = 5,col=1,secondary_y=True)
        fig2.update_xaxes(row = 5,col=1,type='category')

        fig2.add_trace(rateChangeGraph,row = 5,col=2)
        fig2.add_trace(capGraph,row = 6,col=1)
        fig2.update_xaxes(row = 5,col=2,type='category')

        
        fig2.update_layout({"height":3000,
                        "width":1500,
                        "plot_bgcolor":'rgb(28,28,28)',
                        "paper_bgcolor": 'rgb(28,28,28)',
                        "yaxis":{'gridcolor':gridLineColour,'showline':True,'tickfont_size':13,'title_text':'Exposure'},
                        "yaxis2":{'showgrid':False,'showline':True,'tickformat':'.2%','tickfont_size':13,'title_text':'Rate'},
                        "xaxis":{'showgrid':False,'showline':True},
                        "title":{'text':'Risk Factor Graphs','x':0.5,"font_size":22}
                            })
        return fig2