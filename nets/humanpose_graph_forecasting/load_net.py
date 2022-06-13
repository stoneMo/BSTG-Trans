"""
    Utility file to select GraphNN model as
    selected by the user
"""
from nets.humanpose_graph_forecasting.bayesian_st_graph_transformer_net import BayesianSpatioTemporalGraphTransformerNet

def BayesianSpatioTemporalGraphTransformer(net_params, params, config):
    return BayesianSpatioTemporalGraphTransformerNet(net_params, params, config)

def gnn_model(MODEL_NAME, net_params, params, config):
    models = {
        'BayesianSpatioTemporalGraphTransformer': BayesianSpatioTemporalGraphTransformer
    }
        
    return models[MODEL_NAME](net_params, params, config)