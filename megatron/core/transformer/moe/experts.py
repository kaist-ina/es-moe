# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import torch
import nvtx
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import segment_manager
from shared_pinned_memory import upload_experts_params, offload_experts_grads, free_params
shared_pinned_memory = segment_manager.shared_pinned_memory

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.jit import jit_fuser
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe import grouped_gemm_util as gg
from megatron.core.transformer.moe.moe_utils import EsMoeParameter, ExpertPinState
from megatron.core.transformer.transformer_config import TransformerConfig
from time import sleep


class GroupedMLP(MegatronModule):
    """An efficient implementation of the Experts layer using CUTLASS GroupedGEMM.
    
    This class is designed to execute multiple experts in parallel, thereby maximizing computational efficiency.
    """

    def __init__(self, num_local_experts: int, config: TransformerConfig):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.num_local_experts = num_local_experts
        gg.assert_grouped_gemm_is_available()
        assert (
            config.add_bias_linear == False
        ), "bias in the expert layer is not supported in Grouped GEMM yet, please set '--disable-bias-linear' instead."

        self.expert_parallel = config.expert_model_parallel_size > 1
        if self.config.gated_linear_unit:
            if self.config.activation_func != F.silu:
                raise ValueError("Activation function must be silu when using GroupedMLP.")

            @jit_fuser
            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return F.silu(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func

        # How many feature each rank holds for fc1 and fc2, respectively.
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        fc1_output_size = self.config.ffn_hidden_size * self.num_local_experts
        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # (https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
        # and as a result we avoid allocate the transpose of weights.
        # Initialize weight.
        if config.use_cpu_initialization:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight1,
                    self.config.hidden_size,
                    fc1_output_size,
                    fc1_output_size_per_partition,
                    partition_dim=1,
                    init_method=config.init_method,
                    params_dtype=config.params_dtype,
                )
                _initialize_affine_weight_cpu(
                    self.weight2,
                    fc2_input_size,
                    self.config.hidden_size,
                    fc2_input_size_per_partition,
                    partition_dim=0,
                    init_method=config.output_layer_init_method,
                    params_dtype=config.params_dtype,
                )
        else:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight1,
                    config.init_method,
                    partition_dim=1,
                    expert_parallel=self.expert_parallel,
                )
                _initialize_affine_weight_gpu(
                    self.weight2,
                    config.output_layer_init_method,
                    partition_dim=0,
                    expert_parallel=self.expert_parallel,
                )
        setattr(self.weight1, 'allreduce', not self.expert_parallel)
        setattr(self.weight2, 'allreduce', not self.expert_parallel)

    def forward(self, permuted_local_hidden_states, tokens_per_expert):
        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

            fc1_output = gg.ops.gmm(
                permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False
            )

            intermediate_parallel = self.activation_func(fc1_output)

            fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure parameters still have gradients when no tokens are routed to this set of experts.
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            h = self.activation_func(h)
            h = torch.matmul(h, w2)

            fc2_output = h

        return fc2_output, None

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        raise NotImplementedError(
            'Currently distributed checkpointing is not supported for GroupedMLP'
        )


class SequentialMLP(MegatronModule):
    """An implementation of the Experts layer using a sequence of MLP layers.
    
    This class executes each expert sequentially.
    """

    def __init__(self, num_local_experts, config: TransformerConfig, submodules: MLPSubmodules):
        super().__init__(config=config)
        self.add_bias = config.add_bias_linear
        self.num_local_experts = num_local_experts
        self.local_experts = torch.nn.ModuleList()

        # ES-MoE
        self.experts_param_list: List[Dict[Union[Literal['CPU'], Literal['GPU']], List[EsMoeParameter]]] = []
        self._streams: Dict[str, torch.cuda.Stream] = {}
        self._events: Dict[str, List[torch.cuda.Event]] = {}
        self._expert_pin_states: List[ExpertPinState] = [ExpertPinState.UNPINNED for _ in range(self.config.num_moe_experts)]
        self.layer_number: Optional[int] = None
        self.lazy_initialized = False
        self._streams['default'] = torch.cuda.current_stream()
        self._microbatch_bwd_iter_cnt = 0
        self._bwd_ready_grad: List[Dict[str, torch.Tensor]] = [{} for _ in range(self.config.num_moe_experts)]

        
        for exp_id in range(self.config.num_moe_experts):
            expert = MLP(self.config, submodules, is_expert=True)
            self.local_experts.append(expert)

            if self.config.enable_esmoe: 
                for name, param in expert.named_parameters():
                    param.register_hook(partial(self._post_backward_hook, name=name, exp_id=exp_id))

                expert.register_full_backward_pre_hook(partial(self._pre_backward_hook, exp_id=exp_id))

    def forward(self, permuted_local_hidden_states, tokens_per_expert, expert_indices = None):
        with torch.cuda.stream(self._streams["computation"]):

            output_local = torch.zeros_like(permuted_local_hidden_states)
            output_bias_local = None
            if self.add_bias:
                output_bias_local = torch.zeros_like(permuted_local_hidden_states)

            cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
            # Insert zero at the begining for offset index's convenience
            zero_tensor = torch.zeros(1, dtype=torch.long, device=cumsum_num_tokens.device)
            cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))
            for just_index, expert_num in enumerate(expert_indices):
                self._pre_forward_hook(expert_num)
                with nvtx.annotate(f"Forward{expert_num}"):
                    expert = self.local_experts[expert_num]
                    start = cumsum_num_tokens[just_index]
                    end = cumsum_num_tokens[just_index + 1]
                    hidden = permuted_local_hidden_states[start:end]
                    output, output_bias = expert(hidden)
                    assert output.is_cuda, f"Output of expert {expert_num} is not on GPU"

                    output_local[start:end] = output
                    if self.add_bias:
                        output_bias = output_bias.expand_as(output)
                        output_bias_local[start:end, :] = output_bias
                        
                    self._post_forward_hook(expert_num)
        
        self._streams['default'].wait_stream(self._streams["computation"])


        return output_local, output_bias_local

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """ Maps local expert to global experts. """
        sharded_state_dict = {}
        num_global_experts = (
            parallel_state.get_expert_model_parallel_world_size() * self.num_local_experts
        )
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )

        expert_sharded_prefix = f'{prefix}experts.'
        for expert_local_idx, expert in enumerate(self.local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            expert_state_dict_prefix = f'{prefix}local_experts.{expert_local_idx}.'
            expert_sharded_offsets = (
                *sharded_offsets,
                (len(sharded_offsets), expert_global_idx, num_global_experts),
            )

            expert_state_dict = expert.sharded_state_dict(
                expert_state_dict_prefix, expert_sharded_offsets, metadata
            )
            # Remove expert layers indexing from sharded keys
            replace_prefix_for_sharding(
                expert_state_dict, expert_state_dict_prefix, expert_sharded_prefix
            )
            # Adjust replica ids - replication along DP modulo EP
            for k, sh_ten in expert_state_dict.items():
                replica_id = sh_ten.replica_id
                assert (
                    len(replica_id) == 3
                ), f'Expected replica_id for {k} to be in (PP, TP, DP) format, got: {replica_id}'
                sh_ten.replica_id = (
                    *replica_id[:2],
                    parallel_state.get_data_modulo_expert_parallel_rank(),
                )

            sharded_state_dict.update(expert_state_dict)
        return sharded_state_dict

    def init_stream(self) -> None:
        if torch.cuda.is_available():
            self._streams["communication"] = torch.cuda.Stream()
            self._streams["computation"] = torch.cuda.Stream()
            self._streams["post_backward"] = torch.cuda.Stream()

            self._events["comm_forward"] = [torch.cuda.Event() for _ in range(self.config.num_moe_experts)]
            self._events["comm_backward"] = [torch.cuda.Event() for _ in range(self.config.num_moe_experts)]
            self._events["optim_offload"] = [torch.cuda.Event() for _ in range(self.config.num_moe_experts)]
            self._events["optim_upload"] = [torch.cuda.Event() for _ in range(self.config.num_moe_experts)]

    def _lazy_init_param_attributes(self, p: EsMoeParameter, rank: int, layer:int, expert:int, order:int):
        if hasattr(p, "_cpu"):
            return

        original_dtype = p.dtype
        original_device = p.device
        p._cpu = p.data

        # Create pinned memory for the parameter and copy parameter data to pinned memory
        p._cpu = shared_pinned_memory(p.data, rank, layer, expert, order, 1, True)

        # Allocate and free GPU memory for the parameter
        p._gpu = torch.zeros_like(p._cpu, device=original_device, dtype=original_dtype)
        if p._gpu.storage().size() > 0:
            assert p._gpu.storage_offset() == 0
            p._gpu.storage().resize_(0)
        p.data = p._gpu

        p._cpu_grad = shared_pinned_memory(p.data, rank, layer, expert, order, 2, True)
    
    def lazy_init(self, layer_number: int) -> None:
        rank = parallel_state.get_data_parallel_rank()
        with nvtx.annotate("LazyInit"):
            self.layer_number = layer_number
            for exp_id, expert in enumerate(self.local_experts):
                param_dict = {
                    'CPU': [],
                    'GPU': []
                }
                parameters: EsMoeParameter = expert.parameters()
                for param_id, p in enumerate(parameters):
                    self._lazy_init_param_attributes(p, rank, layer_number, exp_id, param_id)
                    param_dict['GPU'].append(p._gpu.data)
                    param_dict['CPU'].append(p._cpu.data)
                self.experts_param_list.append(param_dict)
                
        self.init_stream()
        self.lazy_initialized = True

    @torch.no_grad()
    def _use_gpu_param(self, params: Optional[List[EsMoeParameter]] = None) -> None:
        return
        for p in params:
            p.data = p._gpu

    @torch.no_grad()
    def _use_cpu_param(self, params: Optional[List[EsMoeParameter]] = None) -> None:
        return
        for p in params:
            p.data = p._cpu
    
    @torch.no_grad()
    def _upload_experts(self, exp_id: int, forward = True) -> None:
        with nvtx.annotate("UploadExperts"):
            curr_exp_param_dict = self.experts_param_list[exp_id]

            with torch.cuda.stream(self._streams["communication"]):
                if forward:
                    segment_manager.pre_forward_hook(self.layer_number, exp_id)
                else:
                    segment_manager.pre_backward_hook(self.layer_number, exp_id)

                # will upload parameters in communication stream
                print(f"Uploading GPU {parallel_state.get_expert_model_parallel_rank()} Layer {self.layer_number} EXPERT {exp_id}")
                upload_experts_params(curr_exp_param_dict['CPU'], curr_exp_param_dict['GPU'], self._streams["communication"].cuda_stream)

                if forward:
                    self._events["comm_forward"][exp_id].record(self._streams["communication"])
                else:
                    self._events["comm_backward"][exp_id].record(self._streams["communication"])

    @torch.no_grad()
    def _pre_forward_hook(self, exp_id: int) -> None:
        """
        Pre-forward hook for expert layer. Will wait for comm_forward event recorded at `_upload_experts` function.
        """
        with nvtx.annotate("PreForwardHook"):
            print(f"PreFH: GPU {parallel_state.get_expert_model_parallel_rank()} Layer {self.layer_number} EXPERT {exp_id}")
            self._use_gpu_param(self.local_experts[exp_id].parameters())
            self._streams["computation"].wait_event(self._events["comm_forward"][exp_id])

    @torch.no_grad()
    def _post_forward_hook(self, exp_id: int) -> None:
        with nvtx.annotate("PostForwardHook"):
            # TODO: Use of computation stream here is redundant
            # self._streams["computation"].synchronize()
            self._use_cpu_param(self.local_experts[exp_id].parameters())
            # print(f"PosFH: GPU {parallel_state.get_expert_model_parallel_rank()} Layer {self.layer_number} EXPERT {exp_id}")

    @torch.no_grad()
    def _pre_backward_hook(self, module, grad_in, exp_id: int):
        with nvtx.annotate("PreBackwardHook"):
            # print(f"PreBH: GPU {parallel_state.get_expert_model_parallel_rank()} Layer {self.layer_number} EXPERT {exp_id}")
            self._use_gpu_param(self.local_experts[exp_id].parameters())
            self._upload_experts(exp_id, False) # this will record comm_backward event
            self._streams["computation"].wait_event(self._events["comm_backward"][exp_id])
            self._streams["computation"].wait_stream(self._streams["default"])
        

    @torch.no_grad()
    def _post_backward_hook(self, grad: torch.Tensor, name: str, exp_id: int):
        self._bwd_ready_grad[exp_id][name] = grad
        # print(f"PostBH: GPU {parallel_state.get_expert_model_parallel_rank()} Layer {self.layer_number} EXPERT {exp_id}")
        if len(self._bwd_ready_grad[exp_id]) < len([p for p in self.local_experts[exp_id].parameters() if p.requires_grad]):
            return
        
        with nvtx.annotate("PostBackwardHook"):
            # print(f"PosBH: GPU {parallel_state.get_expert_model_parallel_rank()} Layer {self.layer_number} EXPERT {exp_id}")
            
            self._microbatch_bwd_iter_cnt += 1
            # TODO: offload_experts_grads does not preserve accumulation of gradients. Need to fix this.

            if self._expert_pin_states[exp_id] in [ExpertPinState.UNPINNED, ExpertPinState.UNPINNING]:
                free_params(self.experts_param_list[exp_id]['GPU'], self._streams['computation'].cuda_stream)            
                self._use_cpu_param(self.local_experts[exp_id].parameters())

            self._streams['post_backward'].wait_stream(self._streams["computation"])
            with torch.cuda.stream(self._streams["post_backward"]):
                gpu_grads, cpu_grads = [], []
                param_dict: Dict[str, EsMoeParameter] = {}
                for name, param in self.local_experts[exp_id].named_parameters():
                    param_dict[name] = param

                for name, grad in self._bwd_ready_grad[exp_id].items():
                    cpu_grads.append(param_dict[name]._cpu_grad.data)
                    assert grad is not None,f"{exp_id} {name} does not have gradient"
                    assert grad.is_cuda, f"{exp_id} {name} gradient is not on GPU"
                    gpu_grads.append(grad.data)

                offload_experts_grads(cpu_grads, gpu_grads, self._streams["post_backward"].cuda_stream, 1)
                post_backward_event = torch.cuda.Event()
                post_backward_event.record(self._streams["post_backward"])


            self._streams["default"].wait_stream(self._streams["computation"])
            self._bwd_ready_grad[exp_id].clear()

    @torch.no_grad()
    def wait_for_previous_optim_step(self) -> None:
        
        print("wait_for_previous_optim_step")
        ## NEED TO BE CALLED ON EVERY ITERATION, INCLUDING ACCUM
        for comm_for_event, comm_back_event in zip(self._events["comm_forward"], self._events["comm_backward"]):
            comm_for_event.synchronize()
            comm_back_event.synchronize()