from datetime import datetime
from pathlib import Path
from sklearn.metrics import mean_absolute_error,r2_score
import pandas as pd
import pprint,pickle,pdb,re,json,os
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

pp = pprint.PrettyPrinter(indent=4)

'''model must be a class that implements methods
fit,predict,set_params
'''
class ModelWrapper:
    def __init__(self,model,params={},models_path='./models',scaler=None,unique_name=''):
        self.blueprint = model
        if len(params)>0:
            self.model = model().set_params(**params)
        else:
            self.model = model
        if scaler:
            self.model = Pipeline(steps=[('scaler', scaler()), ('nn', self.model)])

        self.type = re.sub('[^a-z]+','',str(self.model).split('(')[0].lower())
        self.model_name = datetime.now().strftime("%y_%m_%d")+'_'+self.type+unique_name
        self.models_path = Path(models_path)        
        self.features = []
        self.params = params
        self.target = None
        self.scaler = scaler
        self.score='regression'
        
    def top_features(self,X,y,scoring='neg_mean_squared_error'):
        results = permutation_importance(self.model, X,y, scoring=scoring,n_jobs=2,n_repeats=3)
        features = X.columns.tolist()
        res = pd.DataFrame(
            [(features[i],v) for i,v in enumerate(results.importances_mean)],columns=['feature','importance']
        ).sort_values('importance',ascending=False).head(12).set_index('feature')
        res.plot.bar()
        return res
        
    def grid_search(self,data,target,parameters={'alpha':[.7,.8,.9,1], 'l1_ratio':[.4,.5,.6]},scoring='neg_mean_absolute_error',cv=None):
        clf = GridSearchCV(self.model, parameters,scoring=scoring,cv=cv,verbose=1)
        clf.fit(data,target)
        self.params = clf.best_params_
        print(self.params)
        self.model = self.blueprint().set_params(**self.params)
        
    def split(self,data,test=.3):
        n = int(len(data)*(1-test))
        n2 = int(len(data)*test)
        train = data[:n]
        test  = data[n:]
        train2 = data[n2:]
        test2  = data[:n2]        
        return train,test,train2,test2
        
    
    def fit(self,train,train_columns,target_col):
        X = train[train_columns].values
        y = train[target_col].values        
        self.model.fit(X,y)
        self.features = train_columns
        self.target = target_col
    
    def predict(self,data):
        pred = self.model.predict(data[self.features].values)
        return pred
        
    def evaluate(self,data,score='regression'):
        y_hat = self.predict(data[self.features])
        y = data[self.target]
        self.score = score
        self.model_performance = {'target':self.target,'path':str(self.models_path),'params':self.params,'model_name':self.model_name}
        if score=='regression':
            self.model_performance['mae'] = mean_absolute_error(y, y_hat)
            self.model_performance['r2'] = r2_score(y, y_hat)
        self.model_performance['test_length'] = len(y)
        self.model_performance['features'] = len(self.features)
        pp.pprint(self.model_performance)
        self.model_performance['test_date'] = str(datetime.now())
    
    def save(self):
        with open(str(self.models_path/self.model_name)+'.pkl','wb') as fd:
            pickle.dump(self,fd)
        try:
            models = pd.read_csv(self.models_path/'models_performance.csv')
        except:
            models = pd.DataFrame([],columns=self.model_performance.keys())
        models = models.append(self.model_performance,ignore_index=True)
        models.to_csv(self.models_path/'models_performance.csv',index=False)
            
                                 
    def list_models(self):
        res = pd.read_csv(self.models_path/'models_performance.csv')        
        if self.score=='regression':
            res = res.sort_values('mae',ascending=True)
        return res

    
if __name__=="__main__":
    data = pd.read_csv('')
    model = ModelWrapper(model = ml.RandomForestRegressor,params=dict(n_estimators=120, 
                                                                       max_depth = 40,
                                                                       min_samples_split=4,
                                                                       random_state=0))
    train,test,train2,test2 = model.split(data,.2)
    model.fit(train,train_columns=['',''],target_col='your_target_col')
    model.evaluate(test)
    model.save()
    _=model.top_features(test[features],test['Rogowski Input_LV Current'])
    model.grid_search(data[model.features].values,
                      data[model.target].values,
                      parameters = {'n_estimators':[120],
                                    'max_depth':[40],
                                    'min_samples_split':[2,4,6,12],
                                    'max_features':('auto',)},
                      scoring='r2',
                      cv=(
                          [(list(range(len(data))),list(range(len(data))))]
                      )
                     )    
    
        
        
