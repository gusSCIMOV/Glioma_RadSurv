
import os
import json
import glob
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.datasets import load_breast_cancer
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

from sklearn import set_config
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from lifelines.statistics import logrank_test


class DataSurvLoader:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>> Data Loader")
        
    def data_counter(self):
        self.count += 1
        return self.count
    
    def setup_logger(self, **kwargs):
        
        print(f"\n")
        self.method=kwargs['method']
        self.mri_mod=kwargs['mri_mod']
        self.habitat=kwargs['habitats'][kwargs['habitat']]
        self.feature=kwargs['features'][kwargs['feature']]
        self.lambda_l1=kwargs['lambda_l1']
        self.n_alphas=kwargs['n_alphas']
        self.var_threshold=kwargs['var_threshold']
        self.mx_iters=kwargs['mx_iters']
        self.alpha_min=kwargs['alpha_min']
        self.out_folder=kwargs['out_folder']
        

        if len(kwargs['test_ids']) ==1:

            self.site= kwargs['mri_sites'][kwargs['test_ids'][0]]
            self.holdout=self.site
            self.training_sites="_".join([kwargs['mri_sites'][data] for data in kwargs['train_ids']])
        else:
            self.site= kwargs['mri_sites'][kwargs['train_ids'][0]]
            self.holdout= "_".join([kwargs['mri_sites'][data] for data in kwargs['test_ids']])
  
        print(f"Regression method .... {self.method}")
        print(f"Experiment folder .... {self.out_folder}")
        print(f"Main out folder   .... {self.site}")
        print(f"Hold out set      .... {self.holdout}")
        print(f"Training_set      .... {self.training_sites}")
        print(f"MRI modality      .... {self.mri_mod}")
        print(f"Tumor habitat     .... {self.habitat}")
        print(f"Feature familily   ... {self.feature}")
        print(f"{self.method} settings")
        print(f"Variance threshold ... {self.var_threshold}")
        print(f"lambda_L1 ratios   ... {self.lambda_l1}")
        print(f"n_alphas coeff     ... {self.n_alphas}")
        print(f"max num iters      ... {self.mx_iters}")
        print(f"alpha_min ration   ... {self.alpha_min}")
        
        
    def FileTags(self, **kwargs):
        var_threshold=self.var_threshold
        site= self.site
        holdout=self.holdout
        lambda_l1=self.lambda_l1
        mri_mod=self.mri_mod
        habitat=self.habitat
        feature=self.feature

        out_location=os.path.join(self.base_dir,site,self.out_folder,
                                f"training_{self.training_sites}",f"{habitat}_{feature}")

        self.output_base=os.path.join(out_location,'DeepSurv_'+
                                      holdout+'_'+mri_mod+'_varThres'+'_'+str(var_threshold))

        if not os.path.exists(out_location):
            os.makedirs(out_location)
            
        return self.output_base
    
    def get_annotated_csv_paths(self, non_modality=False, clinical_predictors=False, **kwargs):
        required_keys = ['mri_mod','mri_sites', 'time_point', 'tp', 'habitats', 'habitat', 'features', 'feature', 'mat_str']
        
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required argument: {key}")
        
        if not clinical_predictors:
            feat_paths = []
            for data in kwargs['mri_sites']:
                path = os.path.join(
                    self.base_dir, 
                    data, 
                    'Feature_extraction', 
                    'Feature_matrix', 
                    kwargs['time_point'][kwargs["tp"]],
                    f"{data}_{kwargs['mri_mod']}_{kwargs['habitats'][kwargs['habitat']]}_{kwargs['features'][kwargs['feature']]}_{kwargs['mat_str']}"
                )
                if non_modality:
                    path=path.replace(f"{kwargs['mri_mod']}_",'')
                feat_paths.append(path)
        else:
            feat_paths = []
            for data in kwargs['mri_sites']:
                path = os.path.join(
                    self.base_dir, 
                    data, 
                    'Feature_extraction', 
                    'Feature_matrix', 
                    kwargs['time_point'][kwargs["tp"]],
                    f"{data}_{kwargs['features'][kwargs['feature']]}_{kwargs['mat_str']}"
                )
                if non_modality:
                    path=path.replace(f"{kwargs['mri_mod']}_",'')
                feat_paths.append(path)

        return feat_paths
    
    def load_annotated_featuremat(self, csv_paths, train_mat=False, nan_columns_index=None, **kwargs):
        
        dataframes = []
    
        if train_mat:
            print('\nreading train set .... ')
            selected_indices = kwargs['train_ids']
            
            for i in selected_indices:  
                print(csv_paths[i])
                df = pd.read_csv(csv_paths[i])
                dataframes.append(df)
                
            concatenated_df = pd.concat(dataframes, axis=0)
            untrimmed_size = concatenated_df.shape[1]
            
            nan_columns = concatenated_df.columns[concatenated_df.isna().any()].tolist()
            nan_columns_index = concatenated_df.columns.get_indexer(nan_columns)

            sampled_df=concatenated_df.dropna(axis=1)
            trimmed_size = sampled_df.shape[1]
            
            removed_columns = untrimmed_size - trimmed_size

            print(f"Removed {removed_columns} columns containing NaN values in data.")
            print(f"\nresulting training data ... {sampled_df.shape}")

        else:
            print('\nreading testing set .... ')
            selected_indices = kwargs['test_ids']
            for i in selected_indices:  
                print(csv_paths[i])
                df = pd.read_csv(csv_paths[i])
                dataframes.append(df)
                
            concatenated_df = pd.concat(dataframes, axis=0)
            print("trimm NaN columns with training data")
            concatenated_df.drop(concatenated_df.columns[nan_columns_index], axis=1, inplace=True)
            print(f"Removed {len(nan_columns_index)} columns containing NaN values in data.")
            
            if not kwargs['subsampling_test']:
                
                sampled_df=concatenated_df
                print(f"\nresulting testing data ... {sampled_df.shape}")
                
            else:
                print(f"random subsampling test set, {kwargs['val_fraction']} fraction")
                remaining_df, sampled_df=self.dataframe_randsplit(concatenated_df, kwargs)
                print(f"\nresulting testing data ... {sampled_df.shape}")
        
        return sampled_df, nan_columns_index
    
    
    def dataframe_randsplit(self, df, kwargs):
  
        if not 0 < kwargs['val_fraction'] <= 1:
            raise ValueError("sample_fraction must be between 0 and 1")
        
        
        subsample_df, remaining_df = train_test_split(df,test_size=(1 - kwargs['val_fraction']),
                                                       random_state=kwargs['random_state'])
        return remaining_df, subsample_df
    
    
    def load_featuremat(self, df, **kwargs):

        return df.iloc[:, kwargs['feat_onset']:]


    def feature_standardization(self, df, scaler=None, train_mat=False, **kwargs):
        
        if train_mat:
            print('...standardizing training data ...')
            scaler = StandardScaler() # z-score normalization
            scaler.fit(df.iloc[:, kwargs['feat_onset']:])
            Stnd_data = scaler.transform(df.iloc[:, kwargs['feat_onset']:])
        else:
            print('...scaling data with training params ...')
            Stnd_data = scaler.transform(df.iloc[:, kwargs['feat_onset']:])
         
        return Stnd_data, scaler
    
    def survival_annots(self, df1,df2=None,df3=None):
        def pairing_time_events(df, count):
            
            if df is None:
                structured_annots=[]
                patientID=[]
            else:
                splits=['train','val/test','test']
                patientID = df.iloc[:, [0]]
                EventOutcome = df.iloc[:, 1].to_numpy()
                SurvivalTimes = df.iloc[:, 2].to_numpy()
                # Define the dtype for the structured NumPy array
                dtype = [('Event_outcome.tdm', '?'), ('Survival_time.tdm', '<f8')]

                structured_annots = np.zeros(len(EventOutcome), dtype=dtype)
                structured_annots['Event_outcome.tdm'] = EventOutcome
                structured_annots['Survival_time.tdm'] = SurvivalTimes

                uncensored_count = np.sum(EventOutcome)
                censored_count = len(EventOutcome) - uncensored_count
                print(f"{splits[count]} , {len(EventOutcome)} observations ... Uncensored (1) = {uncensored_count}, Censored (0) = {censored_count}")

            return structured_annots, patientID
        
        print("\n .......Survival annotations..........")
        self.count = 0
        structured_annots1,patientID1 = pairing_time_events(df1,self.count)
        structured_annots2,patientID2 = pairing_time_events(df2,self.data_counter())
        structured_annots3,patientID3 = pairing_time_events(df3,self.data_counter())

        return [structured_annots1, structured_annots2, structured_annots3], [patientID1, patientID2, patientID3]
    
    def Varthreshold(self, df_features, train_mat=False, selected_features=None, print_reduction=False):
        
        print("\n... Variance Threshold ...")
        var_threshold=self.var_threshold
        
        if train_mat:
            print("... selecting on training set ...")
            selector = VarianceThreshold(threshold=var_threshold) 
            X_selected = selector.fit_transform(df_features)
            mask = selector.get_support()
            selected_features = df_features.columns[mask]
            df_selected = df_features[selected_features]
            
        else:
            print("... selecting training features on testing set ...")
            df_selected = df_features[selected_features]
            
        
        print("Dimensions original set of features ",df_features.shape)
        print("Dimensions After  variance Threshold {},  var_threshold {}".format(df_selected.shape, var_threshold))

        if print_reduction:
            reduced_sets=[]
            vars_=np.array(range(1,20))/10
            for var_thres in vars_:
                selector = VarianceThreshold(threshold=var_thres)  # Customize threshold
                X_selected = selector.fit_transform(df_features)
                mask = selector.get_support()
                selected_features = df_features.columns[mask]
                Xtrain_selected= df_features[selected_features]
                reduced_sets.append(Xtrain_selected.shape[1])

            plt.plot(vars_,reduced_sets)
            plt.ylabel("Number of features")
            plt.xlabel("variance threshold")

        return df_selected, selected_features 


class CoxNetSurvival:
    def __init__(self, **kwargs):
        self.params = kwargs
        
    def TrainCV(self, Xtrain, Ytrain):
        
        params=self.params
        coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(n_alphas=params['n_alphas'], 
                                                                             alpha_min_ratio=params['alpha_min'], 
                                                                             max_iter=params['mx_iters']))
        coxnet_pipe.fit(Xtrain,Ytrain)
        print(coxnet_pipe)
        print(f"\n>>>>>>>>>>>>>>>> Alpha Discovery :Grid search Train {params['n_splits']} fold-cv and Hold-out test")
        
        self.estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
        
        print(f"\nEstimated_alphas min {self.estimated_alphas[0]} ,  max {self.estimated_alphas[-1]} alphas",)
        cv = KFold(n_splits=params['n_splits'], shuffle=True, random_state=params['random_state'])
        
        self.gridsearch_object = GridSearchCV(
            make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis()),
            param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in self.estimated_alphas],
            "coxnetsurvivalanalysis__l1_ratio": params['lambda_l1']
            },
            cv=cv,
            error_score=0.5,
            n_jobs=params['n_jobs'],
        ).fit(Xtrain, Ytrain)
        

        
        return coxnet_pipe, self.gridsearch_object
    
    def TrainLogger(self,Xtrain, output_base, DataLoader):

        var_threshold=self.params['var_threshold']
        gcv=self.gridsearch_object
        cv_results = pd.DataFrame(gcv.cv_results_)
        alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
        mean = cv_results.mean_test_score
        std = cv_results.std_test_score

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(alphas, mean)
        ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
        ax.set_xscale("log")
        ax.set_ylabel("concordance index")
        ax.set_xlabel("alpha")
        ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
        ax.axhline(0.5, color="grey", linestyle="--")
        ax.grid(True)
    
        plt.savefig(os.path.join(output_base +'_cv_training.png'))
        
        best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
        best_coefs = pd.DataFrame(best_model.coef_, index=Xtrain.columns, columns=["coefficient"])

        non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
        print(f"Number of non-zero coefficients: {non_zero}")

        non_zero_coefs = best_coefs.query("coefficient != 0")
        coef_order = non_zero_coefs.abs().sort_values("coefficient").index

        _, ax = plt.subplots(figsize=(33, 28))
        non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False,fontsize=15)
        ax.set_xlabel("coefficient")
        ax.grid(True)
        plt.savefig(os.path.join(output_base+'_top_features_training.png'))

        self.best_alpha=gcv.best_params_["coxnetsurvivalanalysis__alphas"][0]
        self.best_L1_ratio=gcv.best_params_["coxnetsurvivalanalysis__l1_ratio"]
        training_setup = {
            "Hold out set"  :    DataLoader.holdout,
            "Training_set"  :    DataLoader.training_sites,
            "MRI modality"  :    DataLoader.mri_mod,
            "Tumor habitat" :    DataLoader.habitat,
            "Feature familily":  DataLoader.feature

        }
        train_logs_results = {
            "Train samples": int(Xtrain.shape[0]),
            "Variance Threshold": float(var_threshold), 
            "non_zero coefficients": int(non_zero),  
            "best C-index": float(mean.max()),       
            "Best_alpha_cv": float(self.best_alpha), 
            "Best_L1_ratio_cv": float(self.best_L1_ratio), 
            "max alpha": float(self.estimated_alphas[0]),  
            "min alpha": float(self.estimated_alphas[-1]), 
            "number_of_alphas": int(self.params['n_alphas']),  
            "alpha_min_ratio": float(self.params['alpha_min']), 
            "max_num iter": int(self.params['mx_iters']),  
            "n_splits": int(self.params['n_splits']) 

        }
        # combined_dict = training_setup.copy()  
        # combined_dict.update(train_logs_results)

        combined_dict = {
            "training_setup": training_setup,
            "train_logs_results": train_logs_results
        }

        output_json=output_base +'_cv_train.json'
        
        with open(output_json, 'w') as f:
            json.dump(combined_dict, f, indent=4)
        
        print(f" Results stored in {output_json}")

        best_coefs= best_coefs.sort_values(by='coefficient',ascending=False)
        best_coefs_ = best_coefs[best_coefs['coefficient'] != 0]
        best_coefs_.to_csv(output_base +'_top_featueres.csv', index=True)
        print(f" Results stored in {output_base +'_top_featueres.csv'}")

        return best_model, best_coefs, mean, std
    
    def TrainedModel(self,X_features,Y_annots=None, test=False, coxnet_pred=None):
        
        gcv=self.gridsearch_object
        params=self.params
        
        if not test:
            
            coxnet_pred = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(fit_baseline_model=True))
            coxnet_pred.set_params(**gcv.best_params_)
            best_alpha= self.best_alpha
            coxnet_pred.fit(X_features,Y_annots)
            
            risk_scores = coxnet_pred.predict(X_features)
            self.train_risk_scores = risk_scores
            self.coxnet_pred= coxnet_pred
            self.best_alpha = best_alpha
            
        elif test:
            
            risk_scores = coxnet_pred.predict(X_features)
            self.test_risk_scores = risk_scores

        return risk_scores, coxnet_pred
        
    def Performance_metrics(self, X_features, Y_annots, risk_scores, test=False, risk_threshold=None):
        
        print(f"\nHigh and Low Rsik split")
        
        self.times  = Y_annots['Survival_time.tdm']
        self.events = Y_annots["Event_outcome.tdm"]

        if not test and risk_threshold==None:
            self.median_risk_score = np.median(risk_scores)
            self.below_median = risk_scores <= self.median_risk_score
            self.above_median = risk_scores > self.median_risk_score

            C_harrell_1=self.coxnet_pred.score(X_features,Y_annots)
            print(f"C-index 1: {C_harrell_1}")
            C_harrell_2 = concordance_index_censored(self.events, self.times, risk_scores)
            print(f"C-index 2: {C_harrell_2[0]}")

            log_rank= logrank_test(self.times[self.below_median], self.times[self.above_median], 
                                self.events[self.below_median], self.events[self.above_median])
            
            p_value = log_rank.p_value
            print(f"Log-rank test p-value: {p_value}")
        
            self.train_C_harrell_1=C_harrell_1
            self.train_C_harrell_2=C_harrell_2[0]
            self.train_p_value=p_value

        elif test and risk_threshold!=None:
            self.median_risk_score = risk_threshold
            self.below_median = risk_scores <= self.median_risk_score
            self.above_median = risk_scores > self.median_risk_score

            C_harrell_1=self.coxnet_pred.score(X_features,Y_annots)
            print(f"C-index 1: {C_harrell_1}")
            C_harrell_2 = concordance_index_censored(self.events, self.times, risk_scores)
            print(f"C-index 2: {C_harrell_2[0]}")

            log_rank= logrank_test(self.times[self.below_median], self.times[self.above_median], 
                                self.events[self.below_median], self.events[self.above_median])
            
            p_value = log_rank.p_value
            print(f"Log-rank test p-value: {p_value}")

            self.test_C_harrell_1=C_harrell_1
            self.test_C_harrell_2=C_harrell_2[0]
            self.test_p_value=p_value
        else:
            print(f"invalid args parsing for training OR tesing sets")
            raise ValueError(f"'test' and 'risk_threshold' args must be simultaneosly enabled only for testing set")
        
        return  C_harrell_1, C_harrell_2[0], p_value
    
    def KMSurvival_curves(self, output_base, test=False):
        self.time_below, self.survival_prob_below, ci_below = kaplan_meier_estimator(
            self.events[self.below_median], self.times[self.below_median], conf_type="log-log"
        )
        lower_ci_below, upper_ci_below = ci_below

        self.time_above, self.survival_prob_above, ci_above = kaplan_meier_estimator(
            self.events[self.above_median], self.times[self.above_median], conf_type="log-log"
        )
        lower_ci_above, upper_ci_above = ci_above
        
        
        # Plot the Kaplan-Meier curves with confidence intervals
        plt.figure(figsize=(10, 6))

        plt.step(self.time_below, self.survival_prob_below, where="post", label="Low Risk", color=[.2 ,.8 ,.4])
        plt.fill_between(self.time_below, lower_ci_below, upper_ci_below, step="post", color=[.2 ,.8 ,.4], alpha=0.1)

        plt.step(self.time_above, self.survival_prob_above, where="post", label="High Risk", color="red")
        plt.fill_between(self.time_above, lower_ci_above, upper_ci_above, step="post", color="red", alpha=0.1)

        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
   
        
        if not test:
            plt.title(" Best Alpha {}, C-Index (estimators): Harrell {} , Uno {} . p-value {}".format(
                np.round(self.best_alpha,3),np.round(self.train_C_harrell_1,2),
                np.round(self.train_C_harrell_2, 2),(self.train_p_value)))
            plt.legend()
            plt.grid(True)
            plt.savefig(output_base +'_training_KM.png')
       
        elif test:
            plt.title(" Best Alpha {}, C-Index (estimators): Harrell {} , Uno {} . p-value {}".format(
                np.round(self.best_alpha,3),np.round(self.test_C_harrell_1,2),
                np.round(self.test_C_harrell_2, 2),(self.test_p_value)))
            plt.legend()
            plt.grid(True)
            plt.savefig(output_base +'_testing_KM.png')
        
        plt.show()
            
    def Save_SurvMetrics(self, risk_scores, test_df, output_base):
        
        # saving risk scores as dataframe
        rs_df=pd.DataFrame({'Riskscore': risk_scores})
        df = pd.concat([test_df[['ID','OS_days']],rs_df], axis=1)
        threshold= self.median_risk_score
        df['Risk_label'] = df['Riskscore'].apply(lambda x: 'Low_risk' if x < threshold else 'High_risk')
        df_sorted = df.sort_values(by='Riskscore')
        df_sorted.to_csv(output_base +'_test_riskscores.csv', index=False)

        # saving metrics
        train_results = {
            "risk_scores": [],
            "C1": self.train_C_harrell_1,
            "C2": self.train_C_harrell_2,
            "p-value": self.train_p_value
        }

        test_results = {
            "risk_scores": [],
            "val_fraction": [],
            "Test samples": int(test_df.shape[0]),
            "C1": self.test_C_harrell_1,
            "C2": self.test_C_harrell_2,
            "p-value": self.test_p_value
        }
        
        if self.params['subsampling_test']:
            test_results["val_fraction"] = float(self.params['val_fraction'])
        
        #update dict results     
        surv_results = {
            "train_results": train_results,
            "test_results": test_results
        }

        output_json=output_base +'_results.json'
        
        with open(output_json, 'w') as f:
            json.dump(surv_results, f)
        
        print(f" Results stored in {output_json}")
    
class FeatureConsensus:
    def __init__(self, **kwargs):
        self.params = kwargs  

    def experiment_summary(self,base_dir):
        exp_path=os.paht.join(base_dir,
                            self.params['mri_sites'][self.params['test_ids'][0]],
                            self.params['out_folder'])

        exp_folders=os.listdir(exp_path)
    
        # Filter out only directories
        folders = [entry for entry in exp_folders if os.path.isdir(os.path.join(exp_path, entry))]

        # training json
        for feature in folders:
            self.FeatureSum(self, feature)

        print(folders)


    def FeatureSum(self, feature):
        json_files=glob.glob(os.path.join(feature,'*_cv_train.json'))
        with open(json_files, 'r') as f:
            j_file = json.load(f)
        mri=j_file['training_setup']['MRI modality']
        habitat=j_file['training_setup']['Tumor habitat']
        feature=j_file['training_setup']['Feature familily']

        Var_t=j_file['train_logs_results']['Variance Threshold']
        cv_C_Ind=j_file['train_logs_results']['best C-index']
        cv_n_coeff=j_file['train_logs_results']['non_zero coefficients']
        cv_L1=j_file['train_logs_results']['Best_L1_ratio_cv']
        cv_alpha_m=j_file['train_logs_results']['alpha_min_ratio']
        




