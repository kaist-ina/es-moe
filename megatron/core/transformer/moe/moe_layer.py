# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod

import torch

from megatron.core import parallel_state
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = None):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"
        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router = None
        self.experts = None
        self.token_dispatcher = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        pass

    def set_layer_number(self, layer_number: int):
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        if self.config.moe_grouped_gemm:
            self.experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules, MLPSubmodules)
            self.experts = SequentialMLP(self.num_local_experts, self.config, self.submodules)
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

    def forward(self, hidden_states: torch.Tensor):
        if self.config.enable_esmoe:
            return self.forward_esmoe(hidden_states)

        # process MoE
        scores, indices = self.router(hidden_states)
        (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
            hidden_states, scores, indices
        )
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
        output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
        return output, mlp_bias

    def forward_esmoe(self, hidden_states: torch.Tensor):

        # Lazy Initialize
        if not self.experts.lazy_initialized:
            self.experts.lazy_init(self.layer_number)

        # process MoE
        scores, indices = self.router(hidden_states)
        
        # if parallel_state.get_expert_model_parallel_rank() == 1:
        #     for i in range(indices.shape[0]):
        #         print (indices[i][0].item(), end="")
        #         if i%50==0: print ()
        (dispatched_input, tokens_per_expert, expert_indices) = self.token_dispatcher.token_permutation(
            hidden_states, scores, indices
        )
        # print ("PERM END")

        # print(f"GPU{parallel_state.get_expert_model_parallel_rank()} Layer{self.layer_number} FORWARD START")

        self.experts.wait_for_previous_optim_step()
        # print ("WAIT END")
        event_non_expert_compute = torch.cuda.Event()
        event_non_expert_compute.record()
        for expert_index in expert_indices:
            # print ("IN EXPT UPLOAD LOOP")
            # probably can wait computation stream instead of communication stream
            self.experts._streams["communication"].wait_event(event_non_expert_compute)
            self.experts._upload_experts(expert_index, True)
        # print ("EXPT UPLOAD END")
        # call to self.expert will make the default stream wait for the computation stream to finish
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, expert_indices)
        # if parallel_state.get_expert_model_parallel_rank() == 0:
        #     cnt=0
        #     for i in reversed(range(expert_output.shape[0])):
        #         cnt+=1
        #         if cnt==30: break
        #         print (expert_output[i])
        # if parallel_state.get_expert_model_parallel_rank() == 1:
        #     cnt = 0
        #     for i in reversed(range(expert_output.shape[0])):
        #         cnt+=1
        #         if cnt==400: break
        #         print (expert_output[i])
        #     print ()
        # print (f"GPU {parallel_state.get_expert_model_parallel_rank()} expertOutput {expert_output.shape}")
        # print (f"GPU {parallel_state.get_expert_model_parallel_rank()} bias {mlp_bias.shape}")
        # output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
        # return output, mlp_bias
        #print (f"GPU {parallel_state.get_expert_model_parallel_rank()} BIAS {mlp_bias}")
        # print ("UNPERM START")
        output = self.token_dispatcher.token_unpermutation(expert_output)
        # print ("UNPERM---------")
        mlp_bias = self.token_dispatcher.token_unpermutation(mlp_bias)
        #print (f"GPU {parallel_state.get_expert_model_parallel_rank()} OUTPUT {output}")

        # if parallel_state.get_expert_model_parallel_rank() == 0:
        #     # print (f"output after unperm (shape {output.shape}): {output}")

        # print (f"GPU{parallel_state.get_expert_model_parallel_rank()} Layer{self.layer_number} FORWARD DONE")

        # if parallel_state.get_expert_model_parallel_rank() == 1:
        #     cnt = 0
        #     for i in reversed(range(1024)):
        #         cnt+=1
        #         if cnt==400: break
        #         print (output[1023][7][i])
        #     print ()

                
        return output, mlp_bias