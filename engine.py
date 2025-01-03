from collections import deque

import numpy as np

import torch
from torch import nn

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed import comm as dist
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.pipe.engine import PipelineEngine, TRAIN_BATCH_TIMER, PIPE_SEND_OUTPUT_TIMER, PIPE_SEND_GRAD_TIMER, PIPE_RECV_INPUT_TIMER, PIPE_RECV_GRAD_TIMER
from deepspeed.runtime.pipe import schedule
from deepspeed.runtime.utils import PartitionedTensor
from deepspeed.runtime.activation_checkpointing import checkpointing as ds_checkpointing
from deepspeed.runtime.pipe.module import PipelineModule
from deepspeed.runtime import utils as ds_utils
from deepspeed.runtime.pipe.module import LayerSpec

from utils import count_str, eta_str, log


def initialize(args=None,
               model=None,
               model_parameters=None,
               optimizer=None):
    assert model is not None, "deepspeed.initialize requires a model"

    dist_backend = get_accelerator().communication_backend_name()
    dist.init_distributed(dist_backend=dist_backend)

    config = args.deepspeed_config
    assert config is not None, "DeepSpeed requires --deepspeed_config to specify configuration file"

    mpu = model.mpu()
    config_class = DeepSpeedConfig(config, mpu)
    engine = CustomPipelineEngine(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model_parameters,
        mpu=mpu,
        config=config,
        config_class=config_class
    )

    return engine, engine.optimizer


class CustomPipelineEngine(PipelineEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_steps = None
        self.total_tokens = None
        self.token_counter = None
        self.last_finished = 0
        self.elapsed_hist = [deque(), deque()]
        self.eval_time = None
        self.evals_left = 0


    def train_batch(self):
        if not torch._C.is_grad_enabled():
            raise RuntimeError('train_batch() requires gradients enabled. Use eval_batch() instead.')

        # sequence length may change between macro batches (but not between gradient accumulation steps)
        self.reset_activation_shape()

        self.module.train()
        self._compute_loss = True

        # Do the work
        self.timers(TRAIN_BATCH_TIMER).start()
        sched = schedule.TrainSchedule(micro_batches=self.micro_batches,
                                       stages=self.num_stages,
                                       stage_id=self.stage_id)
        self._exec_schedule(sched)
        agg_losses = self._aggregate_total_losses()
        # Actual training loss is always the first item.
        self.agg_train_loss = agg_losses[0].mean()

        self.timers(TRAIN_BATCH_TIMER).stop()

        if self.global_steps % self.steps_per_print() == 0:
            print(f"+ {self.global_rank}: {self.token_counter.processed_tokens}")
            token_sum = torch.tensor(self.token_counter.processed_tokens, device='cuda')
            dist.reduce(token_sum, 0)
            if self.global_rank == 0:
                eval_rem = self.eval_time * self.evals_left if self.eval_time is not None else 0
                token_sum = token_sum.item() / self.num_stages
                print(f"/ {self.num_stages}\n= {token_sum} (mb={self.micro_batches}, stages={self.num_stages})")
                elapsed = self.timers(TRAIN_BATCH_TIMER).elapsed(reset=True) / 1000.0
                iter_time = elapsed / self.steps_per_print()
                tput = self.train_batch_size() / iter_time
                iter_expr = f'it/s: {1.0/iter_time:.2f}' if iter_time < 1 else f's/it: {iter_time:.3f}'
                tput_expr = f's/sample: {1.0/tput:.2f}' if tput < 1 else f'samples/it: {tput:.3f}'

                if self.last_finished == 0 and self.global_steps > 1:
                    # No ETA for the first step, if it involves a resume as results will be off
                    log(f'step: {self.global_steps:>5} / {self.total_steps:>5} '
                        f'loss: {self.agg_train_loss:0.4f} '
                        f'{iter_expr} '
                        f'{tput_expr}')
                else:
                    processed_tokens = token_sum - self.last_finished
                    tokens_per_second = processed_tokens / elapsed
                    self.elapsed_hist[0].append(processed_tokens)
                    self.elapsed_hist[1].append(iter_time)
                    # we avoid showing ETAs until we have at least 3 data samples as the polyfit is garbage otherwise
                    if len(self.elapsed_hist[0]) > 2:
                        fit = np.polyfit(self.elapsed_hist[0], self.elapsed_hist[1], 1) # , w=weights)
                        # ETA is (1) a constant per batch (i.e. fit[1] * remaining batches), and (2) a linear value that rises with token count (i.e. fit[0] * tokens left)
                        eta = fit[1] * (self.total_steps - self.global_steps) + fit[0] * (self.total_tokens - token_sum)
                        eta += eval_rem
                        eta = f'eta: {eta_str(eta)}'
                    else:
                        fit = None
                        eta = ''

                    while len(self.elapsed_hist[0]) > 30:
                        self.elapsed_hist[0].popleft()
                        self.elapsed_hist[1].popleft()
                    # rolling_eta = sum(self.etas) / len(self.etas)
                    log(f'step: {self.global_steps:>5} / {self.total_steps:>5} '
                        f'loss: {self.agg_train_loss:0.4f} '
                        f'tokens: {int(processed_tokens)} '
                        f'[{count_str(token_sum)} / {count_str(self.total_tokens)} = {100.0 * token_sum/self.total_tokens:.2f}%] '
                        f'tok/s: {tokens_per_second:0.3f} '
                        f'{iter_expr} '
                        f'{tput_expr} '
                        f'{eta}')
                self.last_finished = token_sum
            else:
                self.timers(TRAIN_BATCH_TIMER).elapsed(reset=True)

        # Monitoring
        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [('Train/Samples/train_loss', self.agg_train_loss.mean().item(),
                                    self.global_samples)]
            self.monitor.write_events(self.summary_events)

        if self.wall_clock_breakdown() and self.global_steps % self.steps_per_print() == 0:
            self.timers.log([
                PIPE_SEND_OUTPUT_TIMER,
                PIPE_SEND_GRAD_TIMER,
                PIPE_RECV_INPUT_TIMER,
                PIPE_RECV_GRAD_TIMER,
            ])

        return agg_losses


    def eval_batch(self, data_iter):
        # sequence length may change between macro batches (but not between gradient accumulation steps)
        self.reset_activation_shape()

        self.module.eval()
        self._compute_loss = True

        # Use the provided data iterator
        train_iterator = self.data_iterator
        self.set_dataiterator(data_iter)

        # Do the work
        sched = schedule.InferenceSchedule(micro_batches=self.micro_batches,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id)

        # prevent dead-lock with multiple evals sequence
        dist.barrier()

        with torch.no_grad():
            self._exec_schedule(sched)

        # list of losses
        agg_eval_losses = self._aggregate_total_losses()

        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/eval_loss', agg_eval_losses[0].mean().item(), self.global_samples)]
            self.monitor.write_events(self.summary_events)

        # Restore the training iterator
        self.set_dataiterator(train_iterator)

        return agg_eval_losses


    def _aggregate_total_losses(self):
        all_agg_outputs = []
        # gather each output for all the gradient accumulation steps
        grouped_outputs = [list(x) for x in zip(*self.fwd_outputs)]
        # if any are scalar, make them dim 1 so we can concat across DP ranks
        for outputs in grouped_outputs:
            for i, output in enumerate(outputs):
                if output.dim() == 0:
                    outputs[i] = torch.unsqueeze(output, 0)

        if self.is_last_stage():
            agg_sizes = []
            # loop to gather all the outputs across DP ranks
            for outputs in grouped_outputs:
                # concat all the grad_accum_steps
                concat_outputs = torch.cat(outputs)
                if self.is_data_parallel:
                    # might be different sizes across DP ranks, so, gather all the sizes
                    sizes = [None] * self.grid.get_data_parallel_world_size()
                    torch.distributed.all_gather_object(sizes, concat_outputs.size(), group=self.grid.get_data_parallel_group())
                    # once we know all the sizes we can gather the results across DP ranks
                    gather_result = [torch.zeros(size).to(self.device) for size in sizes]
                    dist.all_gather(gather_result, concat_outputs, group=self.grid.get_data_parallel_group())
                    # and finally, concat
                    agg_output = torch.cat(gather_result)
                else:
                    agg_output = concat_outputs
                agg_sizes.append(agg_output.size())
                all_agg_outputs.append(agg_output)

            # send the sizes, then broadcast to the PP ranks
            if self.is_pipe_parallel:
                torch.distributed.broadcast_object_list([agg_sizes], src=self.global_rank, group=self.grid.get_pipe_parallel_group())
                for agg_output in all_agg_outputs:
                    dist.broadcast(tensor=agg_output, src=self.global_rank, group=self.grid.get_pipe_parallel_group())
        else:
            # get the outputs from the last stage
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
            assert src_rank in self.grid.pp_group
            result = [None]
            torch.distributed.broadcast_object_list(result, src=src_rank, group=self.grid.get_pipe_parallel_group())
            agg_sizes = result[0]
            for agg_size in agg_sizes:
                agg_output = torch.zeros(agg_size).to(self.device)
                dist.broadcast(tensor=agg_output, src=src_rank, group=self.grid.get_pipe_parallel_group())
                all_agg_outputs.append(agg_output)

        return all_agg_outputs


    # We override this to handle the model returning a list of "losses", but only doing backprop on the first.
    def _exec_forward_pass(self, buffer_id):
        self.tput_timer.start()
        self.mem_status('BEFORE FWD', reset_max=True)

        if isinstance(self.pipe_buffers['inputs'][buffer_id], tuple):
            inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][buffer_id])
        else:
            inputs = self.pipe_buffers['inputs'][buffer_id].clone()

        # collect the partitioned input from the previous stage
        if self.is_pipe_partitioned and not self.is_first_stage():
            part_input = PartitionedTensor.from_meta(meta=inputs[0],
                                                     local_part=inputs[1],
                                                     group=self.grid.get_slice_parallel_group())

            inputs = (part_input.full(), *inputs[2:])
            inputs[0].requires_grad = True
            # skip mask
            #inputs[1].requires_grad = True
            part_input = None
            inputs = inputs[0] if len(inputs) == 1 else inputs
            self.pipe_buffers['inputs'][buffer_id] = inputs

        # inputs has no gradient because it is from a cloned tensor
        outputs = super(PipelineEngine, self).forward(inputs)

        # Reset activation checkpointing buffers.
        # Need to call this between evaluation iterations
        if not self.module.training:
            ds_checkpointing.reset()

        # Partition the outputs if we are not the last stage
        if self.is_pipe_partitioned and not self.is_last_stage():
            if isinstance(outputs, tuple):
                first_output = outputs[0]
                # TODO: Improve pipe partitioning to pass multiple tensors that require grads
                assert all([torch.is_tensor(elt) and elt.requires_grad is False for elt in outputs[1:]])
                outputs_tail = outputs[1:]
            elif torch.is_tensor(outputs):
                first_output = outputs
                outputs_tail = []
            else:
                raise ValueError("expecting a tensor or a tuple of tensors")
            part = PartitionedTensor(tensor=first_output, group=self.grid.get_slice_parallel_group())
            # Clear the large output data, but save the computation graph
            first_output.data = torch.zeros(1)
            self.pipe_buffers['output_tensors'][buffer_id] = first_output
            # Inject the partitioned tensor into the output before sending
            outputs = (part.to_meta(), part.data(), *outputs_tail)
            part = None

        self.pipe_buffers['outputs'][buffer_id] = outputs

        # Optionally compute loss on the last device
        if self.is_last_stage():
            if self._compute_loss and self.module.loss_fn is not None:
                labels = self.pipe_buffers['labels'][buffer_id]
                losses = self.module.loss_fn(outputs, labels)
            else:
                # Some models just return loss from forward()
                losses = outputs
            if self.eval_return_logits:
                self.outputs = outputs
            if isinstance(losses, torch.Tensor):
                self.loss = losses
                self.fwd_outputs.append([self.loss.detach()])
            else:
                self.loss = losses[0]
                self.fwd_outputs.append([l.detach() for l in losses])

    # make our forward pass method apply
    PipelineEngine._INSTRUCTION_MAP[schedule.ForwardPass] = _exec_forward_pass


class CustomPipelineModule(PipelineModule):
    def __init__(self, layers, **kwargs):
        super().__init__(layers, **kwargs)

    def _partition_layers(self, method='uniform'):
        num_stages = self._topo.get_dim('pipe')
        stage_id = self._topo.get_coord(self.global_rank).pipe

        if self.global_rank == 0:
            print(f'Partitioning pipeline stages with method {method}')

        method = method.lower()

        estimated_sizes = None
        # Each stage gets a simple uniform number of layers.
        if method == 'uniform':
            num_layers = len(self._layer_specs)
            self.parts = ds_utils.partition_uniform(num_items=num_layers, num_parts=num_stages)
        elif method == 'parameters':
            param_counts = self._count_layer_params()
            self.parts = ds_utils.partition_balanced(weights=param_counts, num_parts=num_stages)
        elif method.startswith('type:'):
            layertype = method.split(':')[1]
            binary_weights = [0] * len(self._layer_specs)
            for idx in self._find_layer_type(layertype):
                binary_weights[idx] = 1
            self.parts = ds_utils.partition_balanced(weights=binary_weights, num_parts=num_stages)
        elif method == 'profile':
            raise NotImplementedError(f'Partitioning method {method} not implemented.')
        elif method == 'estimated_size':
            estimated_sizes = [getattr(l, 'estimated_size', 0) for l in self._layer_specs]
            self.parts = ds_utils.partition_balanced(weights=estimated_sizes, num_parts=num_stages)
        else:
            raise NotImplementedError(f'Partitioning method {method} not implemented.')

        # Print some information on the partitioning.
        if self.global_rank == 0:
            for stage in range(num_stages):
                start = self.parts[stage]
                stop = self.parts[stage + 1]
                print(f'stage={stage} layers={stop - start}')
                for idx, layer in enumerate(self._layer_specs[start:stop]):
                    name = str(layer)
                    if isinstance(layer, LayerSpec):
                        name = layer.typename.__name__
                    if isinstance(layer, nn.Module):
                        name = layer.__class__.__name__
                    else:
                        try:
                            name = layer.__name__
                        except AttributeError:
                            pass
                    logstr = f'    {idx+start:2d}: {name}'
                    if estimated_sizes:
                        es = estimated_sizes[idx+start]
                        logstr += f', estimated size: {es}'
                    print(logstr)
            if self.loss_fn:
                try:
                    print(f'  loss: {self.loss_fn.__name__}')
                except AttributeError:
                    print(f'  loss: {self.loss_fn.__class__.__name__}')
        deepspeed.comm.barrier()

        self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id + 1])
