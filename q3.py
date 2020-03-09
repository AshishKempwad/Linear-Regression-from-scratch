class Airfoil: 
    def _init_(self):
        self.weights=[]
    
    
    def train(self,path):
        import numpy as np
        import pandas as pd 
        dataset=pd.read_csv(path,header=None)
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import r2_score
        
        x=dataset.iloc[:,:-1]
        y=dataset.iloc[:,5]
        weights=[1,1,1,1,1,1]
        weights=np.asarray(weights)
        weights=weights.reshape(len(weights),1)
        self.weights=weights
        scaler=MinMaxScaler()
        x=scaler.fit_transform(x)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3,random_state=42)
        X_train=pd.DataFrame(X_train)
        Bias_array=[1]*(len(X_train))
        X_train.insert(loc=0,column='bias',value=Bias_array)
        X_test=pd.DataFrame(X_test)
        Bias_array1=[1]*(len(X_test))
        X_test.insert(loc=0,column='bias',value=Bias_array1)
        y_test=pd.DataFrame(y_test)
        y_train=pd.DataFrame(y_train)
        y_train=y_train.to_numpy()
        y_test=y_test.to_numpy()
        y_train=y_train.reshape(len(y_train),1)
        y_test=y_test.reshape(len(y_test),1)
        learning_parameter=0.1
        iters=1000
        for i in range(0,iters):
            error=X_train.dot(weights)-y_train
            cost_fun=(X_train.T.dot(error))
            weights=weights-(learning_parameter*cost_fun)*(1/(len(X_train)))
            self.weights=weights
            
    def predict(self,path):
        dataset=pd.read_csv(path,header=None)
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import r2_score
        import numpy as np
        import pandas as pd 
        x=dataset.iloc[:,:-1]
        y=dataset.iloc[:,5]
        scaler=MinMaxScaler()
        x=scaler.fit_transform(x)
        x=pd.DataFrame(x)
        Bias_array=[1]*(len(x))
        x.insert(loc=0,column='bias',value=Bias_array)
        x=x.to_numpy()
        new_weights=self.weights
        predict_labels=x.dot(new_weights)
        return predict_labels
        
        
      

        
        
