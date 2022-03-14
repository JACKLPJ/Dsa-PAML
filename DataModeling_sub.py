import GetModels_sub as GM
class data_modeling:
    def __init__(self,X, y, k, N, add_model_metafeatures, add_data_samples, add_pipeline, address_dmn, verbose):
#         GM.add_model_metafeatures=add_model_metafeatures
#         GM.add_pipeline=add_pipeline
#         GM.add_scaler=add_scaler
#         GM.address_network=address_network     
        #self.add_pipeline=add_pipeline
        self.MODEL=GM.get_model(X, y, k, N, add_model_metafeatures, add_data_samples, add_pipeline, address_dmn,verbose).Model
        #self.MODEL=GM.get_model(X, y, N).Model
        self.result_preprocessor=[]
        self.result_model=[]
        self.verbose = verbose
        for item in self.MODEL[1:]:
            self.result_preprocessor.append(self.pre_processing(item))
            self.result_model.append(self.model(item))
        n=len(self.result_preprocessor)
        if self.verbose:
            print('#######################################')
            for i in range(n):
                print('The data pre_processing method is:  {};'.format(self.result_preprocessor[i]))
                print('The model is:  {}\n'.format(self.result_model[i]))
        self.result=[self.result_preprocessor,self.result_model]  
        if self.verbose:
            print('#######################################')
    def pre_processing(self,dic_config):
        processing_method=dic_config['pre-processor']
        paras={}
        if processing_method=='polynomial':
            paras['degree']=dic_config['polynomial_degree']
            paras['interaction_only']=True if dic_config['polynomial_interaction_only'] else False
            paras['include_bias']=True if dic_config['polynomial_include_bias'] else False
        else:
            paras['n_components']=dic_config['pca_keep_variance']
            paras['whiten']=dic_config['pca_whiten']
        return processing_method,paras
    def model(self,dic_config):
        model_=dic_config['model']
        paras={}
        if model_=='xgradient_boosting':
            paras['colsample_bytree']=dic_config['xgradient_boosting_colsample_bytree']
            paras['colsample_bylevel']=dic_config['xgradient_boosting_colsample_bylevel']
            paras['alpha']=dic_config['xgradient_boosting_reg_alpha']
            # paras['scale_pos_weight']=dic_config['xgradient_boosting_scale_pos_weight']
            paras['learning_rate']=dic_config['xgradient_boosting_learning_rate']
            paras['max_delta_step']=dic_config['xgradient_boosting_max_delta_step']
            paras['base_score']=dic_config['xgradient_boosting_base_score']
            paras['n_estimators']=dic_config['xgradient_boosting_n_estimators']
            paras['subsample']=dic_config['xgradient_boosting_subsample']
            paras['reg_lambda']=dic_config['xgradient_boosting_reg_lambda']
            paras['min_child_weight']=dic_config['xgradient_boosting_min_child_weight']
            paras['max_depth']=dic_config['xgradient_boosting_max_depth']
            paras['gamma']=dic_config['xgradient_boosting_gamma']
        elif model_=='gradient_boosting':
            paras['loss']=dic_config['gradient_boosting_loss']
            paras['max_leaf_nodes']=dic_config['gradient_boosting_max_leaf_nodes']
            paras['learning_rate']=dic_config['gradient_boosting_learning_rate']
            paras['min_samples_leaf']=dic_config['gradient_boosting_min_samples_leaf']
            paras['n_estimators']=dic_config['gradient_boosting_n_estimators']
            paras['subsample']=dic_config['gradient_boosting_subsample']
            paras['min_weight_fraction_leaf']=dic_config['gradient_boosting_min_weight_fraction_leaf']
            paras['max_features']=dic_config['gradient_boosting_max_features']
            paras['min_samples_split']=dic_config['gradient_boosting_min_samples_split']
            paras['max_depth']=dic_config['gradient_boosting_max_depth']
        elif model_=='lda':
            paras['shrinkage']=dic_config['lda_shrinkage']
            if 'lda_shrinkage_factor' in dic_config:
                paras['shrinkage_factor']=dic_config['lda_shrinkage_factor']
            paras['n_components']=min(self.MODEL[0], dic_config['lda_n_components'])
            paras['tol']=dic_config['lda_tol']
        elif model_=='extra_trees':
            paras['bootstrap']=True if dic_config['extra_trees_bootstrap'] else False
            paras['n_estimators']=dic_config['extra_trees_n_estimators']
            paras['min_samples_leaf']=dic_config['extra_trees_min_samples_leaf']
            paras['max_features']=dic_config['extra_trees_max_features'] 
            paras['min_weight_fraction_leaf']=dic_config['extra_trees_min_weight_fraction_leaf']
            paras['criterion']=dic_config['extra_trees_criterion']
            paras['min_samples_split']=dic_config['extra_trees_min_samples_split']
            paras['max_depth']=dic_config['extra_trees_max_depth']
        elif model_=='random_forest':
            paras['bootstrap']=True if dic_config['random_forest_bootstrap'] else False
            paras['n_estimators']=dic_config['random_forest_n_estimators']
            paras['min_samples_leaf']=dic_config['random_forest_min_samples_leaf']
            paras['max_features']=dic_config['random_forest_max_features'] 
            paras['min_weight_fraction_leaf']=dic_config['random_forest_min_weight_fraction_leaf']
            paras['criterion']=dic_config['random_forest_criterion']
            paras['min_samples_split']=dic_config['random_forest_min_samples_split']
            paras['max_depth']=dic_config['random_forest_max_depth']
            paras['max_leaf_nodes']=dic_config['random_forest_max_leaf_nodes']
        elif model_=='decision_tree':
            paras['splitter']=dic_config['decision_tree_splitter']
            paras['min_samples_leaf']=dic_config['decision_tree_min_samples_leaf']
            paras['max_features']=dic_config['decision_tree_max_features'] 
            paras['min_weight_fraction_leaf']=dic_config['decision_tree_min_weight_fraction_leaf']
            paras['criterion']=dic_config['decision_tree_criterion']
            paras['min_samples_split']=dic_config['decision_tree_min_samples_split']
            paras['max_depth']=dic_config['decision_tree_max_depth']
            paras['max_leaf_nodes']=dic_config['decision_tree_max_leaf_nodes']
        elif model_=='libsvm_svc':
            paras['kernel']=dic_config['libsvm_svc_kernel']
            paras['C']=dic_config['libsvm_svc_C']
            paras['max_iter']=dic_config['libsvm_svc_max_iter'] 
            if 'libsvm_svc_degree' in dic_config:
                paras['degree']=dic_config['libsvm_svc_degree']
            if 'libsvm_svc_coef0' in dic_config: 
                paras['coef0']=dic_config['libsvm_svc_coef0']
            paras['tol']=dic_config['libsvm_svc_tol']
            paras['shrinking']=dic_config['libsvm_svc_shrinking']
            paras['gamma']=dic_config['libsvm_svc_gamma']
        elif model_=='k_nearest_neighbors':
            paras['p']=dic_config['k_nearest_neighbors_p']
            paras['weights']=dic_config['k_nearest_neighbors_weights']
            paras['n_neighbors']=dic_config['k_nearest_neighbors_n_neighbors'] 
        elif model_=='bernoulli_nb':
            paras['alpha']=dic_config['bernoulli_nb_alpha'] 
            paras['fit_prior']=True if dic_config['bernoulli_nb_fit_prior'] else False
        elif model_=='multinomial_nb': 
            paras['alpha']=dic_config['multinomial_nb_alpha'] 
            paras['fit_prior']=True if dic_config['multinomial_nb_fit_prior'] else False
        #else:
        elif model_=='qda':
            paras['reg_param']=dic_config['qda_reg_param']
        return model_,paras

