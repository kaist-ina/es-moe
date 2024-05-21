

from typing import Optional
import torch.multiprocessing as mp
import os
import torch
from dataclasses import dataclass
from megatron.core.models.gpt.gpt_layer_specs import _get_mlp_module_spec
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class OptimizerMessage:
    pass

_GLOBAL_ESMOE_OPTIMIZER = None

class EsMoeOptimizer:
    def __init__(self, optimizer_config: OptimizerConfig, transformer_config: TransformerConfig):
        self.optimizer_config = optimizer_config
        self.transformer_config = transformer_config

        self.experts = []
        self.optimizers = []

        self.step_queue: mp.Queue[Optional[OptimizerMessage]] = mp.Queue()
        print("Spawning separate process for ESMoE optimizer")
        self.process = mp.Process(target=self.optimizer_main)
        self.process.start()

    def join(self):
        self.process.join()

    def optimizer_main(self):

        # Set the process to low priority
        os.nice(19)

        # Initialize the optimizer
        optim_cls = torch.optim.Adam
        mlp_spec = _get_mlp_module_spec(
            use_te=True, 
            num_experts=self.transformer_config.num_moe_experts, 
            moe_grouped_gemm=self.transformer_config.moe_grouped_gemm)
        
        for exp_id in range(self.transformer_config.num_moe_experts):
            expert = MLP(self.transformer_config, mlp_spec.submodules, is_expert=True)
            self.experts.append(expert)
            print(expert)

        

        print("HI~")

        while True:
            v = self.step_queue.get() # blocking call

def initialize_esmoe_optimizer(optimizer_config: OptimizerConfig, transformer_config: TransformerConfig):
    _GLOBAL_ESMOE_OPTIMIZER = EsMoeOptimizer(optimizer_config, transformer_config)

def get_global_esmoe_optimizer():
    return _GLOBAL_ESMOE_OPTIMIZER
