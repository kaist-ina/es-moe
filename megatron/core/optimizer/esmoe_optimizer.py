
from enum import Enum
from functools import partial
from queue import Empty
from typing import List, Optional, Set
import torch.multiprocessing
import os
import torch
from dataclasses import dataclass
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.parallel_state import set_tensor_model_parallel_group
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.moe_utils import ParamType

import segment_manager
import pickle
shared_pinned_memory = segment_manager.shared_pinned_memory


mp = torch.multiprocessing.get_context('spawn')

class OptimizerMessageType(Enum):
    OPTIM_STEP = 0
    SCHED_STEP = 1
    UPDATE_CONFIG = 2

@dataclass
class OptimizerMessage:
    message_id: int
    message_type: OptimizerMessageType
    layer_id: int
    expert_id: int
    lr: float = 0
    weight_decay: float = 0



_GLOBAL_ESMOE_OPTIMIZER = None

class EsMoeOptimizer:
    def __init__(self, rank, optimizer_config: OptimizerConfig, transformer_config: TransformerConfig):
        self.optimizer_config = optimizer_config
        self.transformer_config = transformer_config
        self.transformer_config.use_cpu_initialization = True
        self.transformer_config.perform_initialization = False
        for k, v in self.transformer_config.__dict__.items():
            try:
                pickle.dumps(v)
            except AttributeError:
                setattr(self.transformer_config, k, None)
        self.rank = rank

        self.experts: List[List[torch.nn.Module]] = []
        self.optimizers: List[List[torch.optim.Optimizer]] = []

        self.step_queue: torch.multiprocessing.Queue[Optional[OptimizerMessage]] = mp.Queue()
        self.completion_queue: torch.multiprocessing.Queue[Optional[int]] = mp.Queue()
        self.completion_set: Set[int] = set()
        print(f"Spawning separate process for ESMoE optimizer at rank {rank}")
        self.process = mp.Process(target=self.optimizer_main)
        self.process.start()

        self._message_id: int = 0

    @torch.no_grad
    def optimizer_step(self, layer_id, expert_id):
        '''
        Called by host. This a blocking call
        '''
        # print(f"GPU{self.rank} : Optimizer step")
        self._message_id += 1
        msg_id = self._message_id
        # print(f"[{os.getpid()}] Opt L{layer_id} E{expert_id} : Optimizer step requested, message id {msg_id}")
        self.step_queue.put(OptimizerMessage(msg_id, OptimizerMessageType.OPTIM_STEP, layer_id, expert_id))
        self.wait_until_complete(msg_id)
        # print(f"[{os.getpid()}] Opt L{layer_id} E{expert_id} : Optimizer step finished,  message id {msg_id} remaining {self.completion_set}")

    @torch.no_grad
    def scheduler_step(self, lr: float, weight_decay: float):
        '''
        Called by host
        '''
        # print(f"GPU{self.rank} : Scheduler step")
        self._message_id += 1
        self.step_queue.put(OptimizerMessage(-1, OptimizerMessageType.SCHED_STEP, -1, -1, lr=lr, weight_decay=weight_decay))

    def wait_until_complete(self, msg_id: int):
        '''
        Called by host
        '''
        assert msg_id >= 0, "Invalid message id"

        while msg_id not in self.completion_set:
            try:
                v = self.completion_queue.get(timeout=0.001)
                if v >= 0:
                    self.completion_set.add(v)
            except Empty:
                pass
        
        self.completion_set.remove(msg_id)

    def optimizer_main(self):
        
        # tweak
        torch.distributed.init_process_group(backend='gloo', init_method='env://', world_size=1, rank=0)
        set_tensor_model_parallel_group(torch.distributed.new_group([0]))

        from megatron.core.models.gpt.gpt_layer_specs import _get_mlp_module_spec
        from megatron.core.optimizer.DeepSpeedAdam import DeepSpeedCPUAdam

        

        # Set the process to low priority
        os.nice(19)

        # Make mock model + optimizer
        if self.optimizer_config.optimizer == "adam":
            # later change to deepspeed
            optim_cls = partial(DeepSpeedCPUAdam, 
                                lr=self.optimizer_config.lr,
                                betas=(self.optimizer_config.adam_beta1, self.optimizer_config.adam_beta2),
                                eps=self.optimizer_config.adam_eps,
                                weight_decay=self.optimizer_config.weight_decay)

        mlp_spec = _get_mlp_module_spec(
            use_te=True, 
            num_experts=self.transformer_config.num_moe_experts, 
            moe_grouped_gemm=self.transformer_config.moe_grouped_gemm)
        
        for layer_id in range(1, self.transformer_config.num_layers+1): # layer_id starts from 1
            self.experts.append([])
            self.optimizers.append([])
            for exp_id in range(self.transformer_config.num_moe_experts):
                expert = MLP(self.transformer_config, mlp_spec.submodules, is_expert=True)
                self.experts[-1].append(expert)
                for param_id, p in enumerate(expert.parameters()):
                    p.data = shared_pinned_memory(p.data, self.rank, layer_id, exp_id, param_id, ParamType.PARAM, False, True) # skip copy from p.data
                    # if update_freq == 1:
                    p.grad = shared_pinned_memory(p.data, self.rank, layer_id, exp_id, param_id, ParamType.GRAD, False, True) # skip copy from p.data
                

                    assert p.is_cpu
                    assert p.grad.is_cpu
                optimizer = optim_cls(expert.parameters())
                self.optimizers[-1].append(optimizer)

                for group in optimizer.param_groups:
                    for param_id, p in enumerate(group['params']):
                        state = optimizer.state[p]
                        
                        ## make the step shared tensor memory -> modify the deepsped CPU Adam, too
                        state['step'] = shared_pinned_memory(torch.tensor(0.), self.rank, layer_id, exp_id, \
                                                                    param_id, ParamType.OPTIM_STEP, False, True)

                        # gradient momentum
                        state['exp_avg'] = shared_pinned_memory(p.data, self.rank, layer_id, exp_id, \
                                                                    param_id, ParamType.OPTIM_EXP_AVG, False, True)
                        
                        # gradient variances
                        state['exp_avg_sq'] = shared_pinned_memory(p.data, self.rank, layer_id, exp_id, \
                                                                    param_id, ParamType.OPTIM_EXP_AVG_SQ, False, True)
            print(f"GPU{self.rank} : Layer {layer_id} optimizer initialized")

        while True:
            v = self.step_queue.get() # blocking call
            if v is None:
                print(f"Terminating ESMoE optimizer at rank {self.rank}")
                break
        
            if v.message_type == OptimizerMessageType.OPTIM_STEP:
                segment_manager.pre_optimize_hook(v.layer_id, v.expert_id)
                # with nvtx.annotate('step'):
                #     self.optimizers[layer_id][exp_id].step()
                # print(f"GPU{self.rank} : Layer {v.layer_id} expert {v.expert_id} optimizer step")
                self.optimizers[v.layer_id-1][v.expert_id].step()
                segment_manager.post_optimize_hook(v.layer_id, v.expert_id)
                self.completion_queue.put(v.message_id)
            elif v.message_type == OptimizerMessageType.SCHED_STEP:
                for layer_id in range(1, self.transformer_config.num_layers):
                    for exp_id in range(self.transformer_config.num_moe_experts):
                        for param_group in self.optimizers[layer_id-1][exp_id].param_groups:
                            param_group['lr'] = v.lr
                            param_group['weight_decay'] = v.weight_decay
                self.completion_queue.put(v.message_id)
            else:
                raise ValueError(f"Invalid message type {v.message_type}")
        

def initialize_esmoe_optimizer(rank, optimizer_config: OptimizerConfig, transformer_config: TransformerConfig):
    global _GLOBAL_ESMOE_OPTIMIZER
    _GLOBAL_ESMOE_OPTIMIZER = EsMoeOptimizer(rank, optimizer_config, transformer_config)

def get_global_esmoe_optimizer() -> EsMoeOptimizer:
    return _GLOBAL_ESMOE_OPTIMIZER
