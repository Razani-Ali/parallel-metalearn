import torch.nn as nn
import copy
from metalearn.nn_models.basic_layers.BatchNormalization import BatchNorm
from metalearn.nn_models.basic_layers.GRU import *
from metalearn.nn_models.basic_layers.LSTM import *


class ModelTranslator:
    """
    Translates standard PyTorch models into vmap/functional compatible models
    for Meta-Learning pipelines. Automatically replaces standard BatchNorm and 
    RNN layers with custom pure-functional equivalents while preserving all 
    pre-trained weights, biases, and running statistics.
    """

    @staticmethod
    def convert_to_functional(
        model: nn.Module, 
        use_per_step_stats: bool = False, 
        max_inner_steps: int = 5
    ) -> nn.Module:
        """
        Recursively traverses the model and replaces incompatible layers.

        Args:
            model (nn.Module): The standard PyTorch model to convert.
            use_per_step_stats (bool): Whether custom BNs should use MAML++ per-step stats.
            max_inner_steps (int): Max adaptation steps for per-step stats initialization.

        Returns:
            nn.Module: A new model instance with functional-compatible layers.
        """
        # Create a deepcopy to avoid mutating the original model in-place
        functional_model = copy.deepcopy(model)
        
        ModelTranslator._replace_layers(
            functional_model, 
            use_per_step_stats=use_per_step_stats, 
            max_inner_steps=max_inner_steps
        )
        
        return functional_model

    @staticmethod
    def _replace_layers(module: nn.Module, use_per_step_stats: bool, max_inner_steps: int):
        """
        Recursive helper function to find and replace target layers.
        """
        for name, child in module.named_children():
            
            # ==========================================
            # 1. Batch Normalization Translation
            # ==========================================
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                custom_bn = BatchNorm(
                    num_features=child.num_features,
                    eps=child.eps,
                    momentum=child.momentum,
                    track_running_stats=child.track_running_stats,
                    use_per_step_stats=use_per_step_stats,
                    max_inner_steps=max_inner_steps,
                    learn_gamma=child.affine,
                    learn_beta=child.affine
                )

                # Transfer pre-trained weights (Gamma / Beta)
                if child.affine:
                    if use_per_step_stats:
                        # Expand 1D weights to 2D (Step, Channel)
                        custom_bn.weight.data[:] = child.weight.data.unsqueeze(0)
                        custom_bn.bias.data[:] = child.bias.data.unsqueeze(0)
                    else:
                        custom_bn.weight.data.copy_(child.weight.data)
                        custom_bn.bias.data.copy_(child.bias.data)

                # Transfer Running Statistics (Mean / Variance)
                if child.track_running_stats:
                    if use_per_step_stats:
                        custom_bn.running_mean.data[:] = child.running_mean.data.unsqueeze(0)
                        custom_bn.running_var.data[:] = child.running_var.data.unsqueeze(0)
                    else:
                        custom_bn.running_mean.data.copy_(child.running_mean.data)
                        custom_bn.running_var.data.copy_(child.running_var.data)

                # Replace the layer in the parent module
                setattr(module, name, custom_bn)

            # ==========================================
            # 2. GRU Translation
            # ==========================================
            elif isinstance(child, nn.GRU):
                assert child.num_layers == 1, "Translator currently supports 1-layer RNNs. Stack them manually if needed."
                
                if child.bidirectional:
                    custom_gru = ParallelBiGRU(input_dim=child.input_size, hidden_dim=child.hidden_size)
                    
                    # Forward Direction (Index 0)
                    custom_gru.w_ih.data[0].copy_(child.weight_ih_l0.data)
                    custom_gru.w_hh.data[0].copy_(child.weight_hh_l0.data)
                    # Backward Direction (Index 1)
                    custom_gru.w_ih.data[1].copy_(child.weight_ih_l0_reverse.data)
                    custom_gru.w_hh.data[1].copy_(child.weight_hh_l0_reverse.data)
                    
                    if child.bias:
                        custom_gru.bias_ih.data[0].copy_(child.bias_ih_l0.data)
                        custom_gru.bias_hh.data[0].copy_(child.bias_hh_l0.data)
                        custom_gru.bias_ih.data[1].copy_(child.bias_ih_l0_reverse.data)
                        custom_gru.bias_hh.data[1].copy_(child.bias_hh_l0_reverse.data)
                        
                    setattr(module, name, custom_gru)
                else:
                    custom_gru = UniGRU(input_dim=child.input_size, hidden_dim=child.hidden_size)
                    custom_gru.w_ih.data.copy_(child.weight_ih_l0.data)
                    custom_gru.w_hh.data.copy_(child.weight_hh_l0.data)
                    if child.bias:
                        custom_gru.bias_ih.data.copy_(child.bias_ih_l0.data)
                        custom_gru.bias_hh.data.copy_(child.bias_hh_l0.data)
                    setattr(module, name, custom_gru)

            # ==========================================
            # 3. LSTM Translation
            # ==========================================
            elif isinstance(child, nn.LSTM):
                assert child.num_layers == 1, "Translator currently supports 1-layer RNNs. Stack them manually if needed."
                
                if child.bidirectional:
                    custom_lstm = ParallelBiLSTM(input_dim=child.input_size, hidden_dim=child.hidden_size)
                    
                    # Forward Direction (Index 0)
                    custom_lstm.w_ih.data[0].copy_(child.weight_ih_l0.data)
                    custom_lstm.w_hh.data[0].copy_(child.weight_hh_l0.data)
                    # Backward Direction (Index 1)
                    custom_lstm.w_ih.data[1].copy_(child.weight_ih_l0_reverse.data)
                    custom_lstm.w_hh.data[1].copy_(child.weight_hh_l0_reverse.data)
                    
                    if child.bias:
                        custom_lstm.bias_ih.data[0].copy_(child.bias_ih_l0.data)
                        custom_lstm.bias_hh.data[0].copy_(child.bias_hh_l0.data)
                        custom_lstm.bias_ih.data[1].copy_(child.bias_ih_l0_reverse.data)
                        custom_lstm.bias_hh.data[1].copy_(child.bias_hh_l0_reverse.data)
                        
                    setattr(module, name, custom_lstm)
                else:
                    custom_lstm = UniLSTM(input_dim=child.input_size, hidden_dim=child.hidden_size)
                    custom_lstm.w_ih.data.copy_(child.weight_ih_l0.data)
                    custom_lstm.w_hh.data.copy_(child.weight_hh_l0.data)
                    if child.bias:
                        custom_lstm.bias_ih.data.copy_(child.bias_ih_l0.data)
                        custom_lstm.bias_hh.data.copy_(child.bias_hh_l0.data)
                    setattr(module, name, custom_lstm)

            # ==========================================
            # 4. Recursive Traversal for Nested Modules
            # ==========================================
            else:
                # If it's a Sequential, ModuleList, or Custom Block, dive deeper
                ModelTranslator._replace_layers(child, use_per_step_stats, max_inner_steps)

## Example Usage
# import torchvision.models as models

# standard_resnet = models.resnet18(pretrained=True)

# functional_resnet = ModelTranslator.convert_to_functional(
#     model=standard_resnet, 
#     use_per_step_stats=True, 
#     max_inner_steps=5
# )

# meta_model = MAML_Model(backbone=functional_resnet, head=...)