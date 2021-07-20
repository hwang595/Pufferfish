import datetime
import os
import time

import numpy as np
import torch

try:
    import bit2byte
except ImportError:
    pass


def n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()


def all_reduce(*args, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_reduce(*args, **kwargs)


def all_gather(out_list, in_tensor, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_gather(out_list, in_tensor, **kwargs)
    else:
        assert len(out_list) == 1
        out_list[0].data = in_tensor


class TensorBuffer():
    """
    Packs multiple tensors into one flat buffer for efficient
    intra-worker communication.
    """
    def __init__(self, tensors):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._tensors = tensors

        self.buffer = torch.cat([t.view(-1) for t in tensors]) # copies
    
    def __getitem__(self, index):
        return self.buffer[self._start_idx[index] : self._end_idx[index]].view(*self._tensors[index].shape)

    def __len__(self):
        return len(self._tensors)

    def pack(self, tensors=None):
        # Optional. init already does this.
        if tensors is None:
            tensors = self._tensors
        for tensor, entry in zip(tensors, self):
            entry[:] = tensor

    def unpack(self, tensors):
        for tensor, entry in zip(tensors, self):
            tensor[:] = entry

    def nelement(self):
        return self.buffer.nelement()

    def element_size(self):
        return self.buffer.element_size()

    def bits(self):
        return 8 * self.nelement() * self.element_size()

    def all_reduce(self, async_op=False):
        return torch.distributed.all_reduce(self.buffer, async_op=async_op)
    
    def all_gather(self, async_op=False):
        n_workers = torch.distributed.get_world_size() if torch.distributed.is_available() else 1
        buffers = [torch.empty_like(self.buffer) for i in range(n_workers)]
        handle = all_gather(buffers, self.buffer, async_op=async_op)
        if async_op:
            return buffers, handle
        else:
            return buffers


class Reducer:
    def __init__(self, random_seed, device):
        self.rng = np.random.RandomState(random_seed)
        M = 1024 * 1024
        self.precalc_numbers = (
            torch.from_numpy(self.rng.randn(128 * M)).to(device).type(torch.float32)
        )
        if torch.distributed.is_available():
            self.n_workers = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.n_workers = 1
            self.rank = 0
        self.device = device

    def reduce(self, grad_in, grad_out, memory_out):
        """Return communicated bits"""
        raise NotImplementedError()


class ReducerSignleNode:
    def __init__(self, random_seed, device):
        self.rng = np.random.RandomState(random_seed)
        M = 1024 * 1024
        self.precalc_numbers = (
            torch.from_numpy(self.rng.randn(128 * M)).to(device).type(torch.float32)
        )
        self.n_workers = 1
        self.rank = 0
        self.device = device

    def reduce(self, grad_in, grad_out, memory_out):
        """Return communicated bits"""
        raise NotImplementedError()


class SignCompressor:
    """Taken from https://github.com/PermiJW/signSGD-with-Majority-Vote"""

    def packing(self, src_tensor):
        src_tensor = torch.sign(src_tensor)
        src_tensor_size = src_tensor.size()
        src_tensor = src_tensor.view(-1)
        src_len = len(src_tensor)
        add_elm = 32 - (src_len % 32)
        if src_len % 32 == 0:
            add_elm = 0
        new_tensor = torch.zeros([add_elm], dtype=torch.float32, device=src_tensor.device)
        src_tensor = torch.cat((src_tensor, new_tensor), 0)
        src_tensor = src_tensor.view(32, -1)
        src_tensor = src_tensor.to(dtype=torch.int32)
        dst_tensor = bit2byte.packing(src_tensor)
        dst_tensor = dst_tensor.to(dtype=torch.int32)
        return dst_tensor, src_tensor_size

    def unpacking(self, src_tensor, src_tensor_size):
        src_element_num = self.element_num(src_tensor_size)
        add_elm = 32 - (src_element_num % 32)
        if src_element_num % 32 == 0:
            add_elm = 0
        src_tensor = src_tensor.int()
        new_tensor = torch.ones(
            src_element_num + add_elm, device=src_tensor.device, dtype=torch.int32
        )
        new_tensor = new_tensor.view(32, -1)
        new_tensor = bit2byte.unpacking(src_tensor, new_tensor)
        new_tensor = new_tensor.view(-1)
        new_tensor = new_tensor[:src_element_num]
        new_tensor = new_tensor.view(src_tensor_size)
        new_tensor = -new_tensor.add_(-1)
        new_tensor = new_tensor.float()
        return new_tensor

    def majority_vote(self, src_tensor_list):
        voter_num = len(src_tensor_list)
        src_tensor = torch.stack(src_tensor_list)
        src_tensor = src_tensor.view(-1)
        full_size = 32 * len(src_tensor)
        new_tensor = torch.ones(full_size, device=src_tensor.device, dtype=torch.int32)
        new_tensor = new_tensor.view(32, -1)
        new_tensor = bit2byte.unpacking(src_tensor, new_tensor)
        new_tensor = -new_tensor.add_(-1)
        # sum
        new_tensor = new_tensor.permute(1, 0).contiguous().view(voter_num, -1)
        new_tensor = torch.sum(new_tensor, 0)
        new_tensor = new_tensor.view(-1, 32).permute(1, 0)
        new_tensor = torch.sign(new_tensor)
        new_tensor = bit2byte.packing(new_tensor)
        new_tensor = new_tensor.to(dtype=torch.int32)
        return new_tensor

    def element_num(self, size):
        num = 1
        for i in range(len(size)):
            num *= size[i]
        return num

    def compress(self, src_tensor):
        return self.packing(src_tensor)

    def uncompress(self, src_tensor, src_tensor_size):
        dst_tensor = self.unpacking(src_tensor, src_tensor_size)
        return dst_tensor


class SignSGDwithMajorityVoteReducer(Reducer):
    def reduce(self, aggregated_grad):
        """
        Reduce gradients between the workers in place
        :aggregated_grad: here we actually compress momentum rather than gradient
        """
        enc_start = torch.cuda.Event(enable_timing=True)
        enc_end = torch.cuda.Event(enable_timing=True)
        comm_start = torch.cuda.Event(enable_timing=True)
        comm_end = torch.cuda.Event(enable_timing=True)
        dec_start = torch.cuda.Event(enable_timing=True)
        dec_end = torch.cuda.Event(enable_timing=True)

        bits_communicated = 0
        comm_time_counter = 0
        encode_decode_counter = 0

        # encoding stage
        enc_start.record()
        sign_compressor = SignCompressor()
        #with self.timer("reduce.flatpack"):
        #flatgrad = TensorBuffer(grad_in)

        #with self.timer("reduce.compress", verbosity=2):
        my_bits, sign_size = sign_compressor.compress(aggregated_grad)

        #with self.timer("reduce.gather", verbosity=2): # let's always assume we are with distributed mode
        bits = [torch.empty_like(my_bits) for i in range(self.n_workers)]
        enc_end.record()
        torch.cuda.synchronize()
        enc_cost = float(enc_start.elapsed_time(enc_end)) / 1000.0
        encode_decode_counter += enc_cost

        comm_start.record()
        all_gather(bits, my_bits)
        comm_end.record()
        torch.cuda.synchronize()
        comm_time_counter = float(comm_start.elapsed_time(comm_end)) / 1000.0 # in seconds

        # decoding stage
        dec_start.record()
        bits_communicated += n_bits(my_bits)  # for the norm vector, being optimistic here

        #with self.timer("reduce.decompress", verbosity=2):
        sum_of_signs = None
        for their_bits in bits:
            uncompressed = sign_compressor.uncompress(their_bits, sign_size)
            if sum_of_signs is None:
                sum_of_signs = uncompressed
            else:
                sum_of_signs += uncompressed

        #with self.timer("reduce.majorityvote", verbosity=2):
        #total_sign = sum_of_signs.sign()
        reduced_aggregated_grad = sum_of_signs.sign()

        dec_end.record()
        torch.cuda.synchronize()
        dec_cost = float(dec_start.elapsed_time(dec_end)) / 1000.0
        encode_decode_counter += dec_cost
        #with self.timer("reduce.set_out", verbosity=2):
        #flatgrad.buffer = total_sign
        #for out, majorityvote in zip(grad_out, flatgrad):
        #    out.data[:] = majorityvote

        # no error feedback scheme is used here, this we do not enable the 
        #with self.timer("reduce.memory", verbosity=2):
        #for mem in memory_out:
        #    mem.data[:] = -10_000_000  # don't try to use memory
        #return bits_communicated
        return reduced_aggregated_grad, bits_communicated, comm_time_counter, encode_decode_counter


class SignSGDwithMajorityVoteReducerSimulation(Reducer):
    def reduce(self, aggregated_grad):
        """
        Reduce gradients between the workers in place
        :aggregated_grad: here we actually compress momentum rather than gradient
         and we do aggregated gradient # simulated nodes X dimension
        """
        enc_start = torch.cuda.Event(enable_timing=True)
        enc_end = torch.cuda.Event(enable_timing=True)
        comm_start = torch.cuda.Event(enable_timing=True)
        comm_end = torch.cuda.Event(enable_timing=True)
        dec_start = torch.cuda.Event(enable_timing=True)
        dec_end = torch.cuda.Event(enable_timing=True)

        bits_communicated = 0
        comm_time_counter = 0
        encode_decode_counter = 0

        # encoding stage
        enc_start.record()
        sign_compressor = SignCompressor()

        num_simulated_nodes = aggregated_grad.size()[0]

        my_bits_list = []
        sign_size_list = []
        for i in range(num_simulated_nodes):
            my_bits, sign_size = sign_compressor.compress(aggregated_grad[i])
            my_bits_list.append(my_bits)
            sign_size_list.append(sign_size)

        my_bits_list = torch.stack(my_bits_list) # # of simulated nodes X compressed tensors     

        bits = [torch.empty_like(my_bits_list) for i in range(self.n_workers)]
        enc_end.record()
        torch.cuda.synchronize()
        enc_cost = float(enc_start.elapsed_time(enc_end)) / 1000.0
        encode_decode_counter += enc_cost

        comm_start.record()
        all_gather(bits, my_bits_list)
        comm_end.record()
        torch.cuda.synchronize()
        comm_time_counter = float(comm_start.elapsed_time(comm_end)) / 1000.0 # in seconds

        # decoding stage
        dec_start.record()
        bits_communicated += n_bits(my_bits_list)  # for the norm vector, being optimistic here

        # unroll the bits
        unrolled_bits = []
        for their_bits in bits:
            for node_index in range(num_simulated_nodes):
                unrolled_bits.append(their_bits[node_index])

        sum_of_signs = None
        for their_bits in unrolled_bits:
            uncompressed = sign_compressor.uncompress(their_bits, sign_size)
            if sum_of_signs is None:
                sum_of_signs = uncompressed
            else:
                sum_of_signs += uncompressed

        reduced_aggregated_grad = sum_of_signs.sign()

        dec_end.record()
        torch.cuda.synchronize()
        dec_cost = float(dec_start.elapsed_time(dec_end)) / 1000.0
        encode_decode_counter += dec_cost
        return reduced_aggregated_grad, bits_communicated, comm_time_counter, encode_decode_counter


class SignSGDwithMajorityVoteReducerSimulationSignleNode(ReducerSignleNode):
    def reduce(self, aggregated_grad):
        """
        Reduce gradients between the workers in place
        :aggregated_grad: here we actually compress momentum rather than gradient
         and we do aggregated gradient # simulated nodes X dimension
        """
        enc_start = torch.cuda.Event(enable_timing=True)
        enc_end = torch.cuda.Event(enable_timing=True)
        comm_start = torch.cuda.Event(enable_timing=True)
        comm_end = torch.cuda.Event(enable_timing=True)
        dec_start = torch.cuda.Event(enable_timing=True)
        dec_end = torch.cuda.Event(enable_timing=True)

        bits_communicated = 0
        comm_time_counter = 0
        encode_decode_counter = 0

        # encoding stage
        enc_start.record()
        sign_compressor = SignCompressor()

        num_simulated_nodes = aggregated_grad.size()[0]

        my_bits_list = []
        sign_size_list = []
        
        for i in range(num_simulated_nodes):
            my_bits, sign_size = sign_compressor.compress(aggregated_grad[i])
            my_bits_list.append(my_bits)
            sign_size_list.append(sign_size)

        my_bits_list = torch.stack(my_bits_list) # # of simulated nodes X compressed tensors     

        #bits = [torch.empty_like(my_bits_list) for i in range(self.n_workers)]
        enc_end.record()
        torch.cuda.synchronize()
        enc_cost = float(enc_start.elapsed_time(enc_end)) / 1000.0
        encode_decode_counter += enc_cost

        # comm_start.record()
        # #all_gather(bits, my_bits_list)
        # comm_end.record()
        # torch.cuda.synchronize()
        comm_time_counter = 0 # in seconds

        # decoding stage
        dec_start.record()
        bits_communicated += n_bits(my_bits_list)  # for the norm vector, being optimistic here

        # unroll the bits
        #unrolled_bits = []
        #for their_bits in my_bits_list:
        #    for node_index in range(num_simulated_nodes):
        #        unrolled_bits.append(their_bits[node_index])

        sum_of_signs = None
        #for their_bits, their_sign_size in zip(my_bits_list, sign_size_list):
        #    uncompressed = sign_compressor.uncompress(their_bits, their_sign_size)
        for their_bits in my_bits_list:
            uncompressed = sign_compressor.uncompress(their_bits, sign_size)
            if sum_of_signs is None:
                sum_of_signs = uncompressed
            else:
                sum_of_signs += uncompressed

        reduced_aggregated_grad = sum_of_signs.sign()

        dec_end.record()
        torch.cuda.synchronize()
        dec_cost = float(dec_start.elapsed_time(dec_end)) / 1000.0
        encode_decode_counter += dec_cost
        return reduced_aggregated_grad, bits_communicated, comm_time_counter, encode_decode_counter


class StochasticUniformQuantization(Reducer):

    def reduce(self, aggregated_grad):
        """
        implement the binary stochastic quantization in https://arxiv.org/pdf/1611.00429.pdf
        """
        bits_communicated = 0
        #sign_compressor = SignCompressor()
        comm_time_counter = 0
        encode_decode_counter = 0

        quantization_start = torch.cuda.Event(enable_timing=True)
        quantization_end = torch.cuda.Event(enable_timing=True)
        
        quantization_start.record()
        min_val = torch.min(aggregated_grad)
        max_val = torch.max(aggregated_grad)
        prob_vector = (aggregated_grad - min_val)/(max_val-min_val)
        dices_vector = torch.bernoulli(prob_vector) # binary
        dices_vector = dices_vector * 2 - 1
        min_max_val = torch.zeros(2).to(self.device)
        min_max_val[0] = min_val
        min_max_val[1] = max_val
        compressor = SignCompressor()
        
        quantization_end.record()
        torch.cuda.synchronize()
        
        # initialize the min max val tensor
        encode_start = torch.cuda.Event(enable_timing=True)
        encode_end = torch.cuda.Event(enable_timing=True)
        #min_max_val = torch.stack((min_val, max_val), dim=0)
        #print("min max val size: {}, min max val device: {}".format(min_max_val.size(), min_max_val.device))
        encode_start.record()
        compressed_dice, dice_size = compressor.compress(dices_vector)
        encode_end.record()
        torch.cuda.synchronize()

        comm_start = torch.cuda.Event(enable_timing=True)
        comm_end = torch.cuda.Event(enable_timing=True)
        # two all-gather to collect
        comm_start.record()
        gathered_buffer_grad = [torch.empty_like(compressed_dice) for i in range(self.n_workers)]
        torch.distributed.all_gather(gathered_buffer_grad, compressed_dice)

        gathered_buffer_min_max = [torch.empty_like(min_max_val) for i in range(self.n_workers)]
        torch.distributed.all_gather(gathered_buffer_min_max, min_max_val)

        bits_communicated += n_bits(compressed_dice)  # for the norm vector, being optimistic here
        bits_communicated += n_bits(min_max_val)
        comm_end.record()
        torch.cuda.synchronize()


        decode_start = torch.cuda.Event(enable_timing=True)
        decode_end = torch.cuda.Event(enable_timing=True) 
        decode_start.record()
        # decompression
        unquantized_aggregated_grad = []
        for worker_index in range(self.n_workers):
            gathered_buffer_grad[worker_index] = compressor.uncompress(gathered_buffer_grad[worker_index], 
                                                    aggregated_grad.size())
            gathered_buffer_grad[worker_index] = (gathered_buffer_grad[worker_index] + 1) / 2

            #tempt_grad = torch.where(gathered_buffer_grad[worker_index]==1, max_val, min_val)
            unquantized_aggregated_grad.append(torch.where(gathered_buffer_grad[worker_index]==1, 
                                                            gathered_buffer_min_max[worker_index][1],
                                                            gathered_buffer_min_max[worker_index][0]))
                                                            #max_val, min_val))
            #unquantized_aggregated_grad.append(gathered_buffer_grad[worker_index])
            #unquantized_aggregated_grad.append(tempt_grad)

        reduced_aggregated_grad = torch.stack(unquantized_aggregated_grad, dim=0).sum(dim=0)
        decode_end.record()
        torch.cuda.synchronize()
        #print("reduced_aggregated_grad: {}".format(reduced_aggregated_grad))
        print("Time Elapsed quan: {} encode: {}, comm: {}, decode: {}".format(quantization_start.elapsed_time(quantization_end),
                                                            encode_start.elapsed_time(encode_end),
                                                            comm_start.elapsed_time(comm_end),
                                                            decode_start.elapsed_time(decode_end)))
        comm_time_counter = float(comm_start.elapsed_time(comm_end)) / 1000.0 # in seconds
        encode_decode_counter = (float(quantization_start.elapsed_time(quantization_end))+encode_start.elapsed_time(encode_end)+decode_start.elapsed_time(decode_end))/1000.0
        return reduced_aggregated_grad, bits_communicated, comm_time_counter, encode_decode_counter



class RankKReducer(Reducer):
    def __init__(self, random_seed, device, n_power_iterations=0, reuse_query=False, rank=1):
        super().__init__(random_seed, device)
        assert n_power_iterations == 0
        self.rank = rank
        self.p_memory = None
        self.q_memory = None
        self.reuse_query = reuse_query

    def set_random(self, vector):
        torch.manual_seed(self.rng.randint(1_000_000_000))
        vector.data[:] = torch.randn(*vector.shape, device=self.device)
        # orthogonalize(vector)

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        comm_time_counter = 0
        encode_decode_counter = 0

        enc_start_allocate_memory = torch.cuda.Event(enable_timing=True)
        enc_end_allocate_memory = torch.cuda.Event(enable_timing=True)
        enc_start_prepare_q = torch.cuda.Event(enable_timing=True)
        enc_end_prepare_q = torch.cuda.Event(enable_timing=True)
        enc_start_compute_p = torch.cuda.Event(enable_timing=True)
        enc_end_compute_p = torch.cuda.Event(enable_timing=True)
        enc_start_r1_pack = torch.cuda.Event(enable_timing=True)
        enc_end_r1_pack = torch.cuda.Event(enable_timing=True)
        enc_start_normalize_p = torch.cuda.Event(enable_timing=True)
        enc_end_normalize_p = torch.cuda.Event(enable_timing=True)
        enc_start_compute_q = torch.cuda.Event(enable_timing=True)
        enc_end_compute_q = torch.cuda.Event(enable_timing=True)

        comm_start_reduce_p = torch.cuda.Event(enable_timing=True)
        comm_end_reduce_p = torch.cuda.Event(enable_timing=True)
        comm_start_rank1_allreduce = torch.cuda.Event(enable_timing=True)
        comm_end_rank1_allreduce = torch.cuda.Event(enable_timing=True)        
        comm_start_reduce_q = torch.cuda.Event(enable_timing=True)
        comm_end_reduce_q = torch.cuda.Event(enable_timing=True)

        dec_start_outerprod = torch.cuda.Event(enable_timing=True)
        dec_end_outerprod = torch.cuda.Event(enable_timing=True)
        dec_start_r1_unpack = torch.cuda.Event(enable_timing=True)
        dec_end_r1_unpack = torch.cuda.Event(enable_timing=True)

        # Split the tensors into rank1-ones that will be reduced un-compressed
        # and rank > 1 tensors that are compressed
        enc_start_allocate_memory.record()
        rank1_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() <= 1
        ]
        high_rank_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() > 1
        ]

        # We are building a rank-1 approximation of every tensor
        # that can be interpreted as a matrix. Let the approximation be
        # M = p q^T
        # We are allocating consequtive memory for the p's and q's
        memory_is_uninitialized = self.p_memory is None

        #with self.timer("reduce.allocate_memory", verbosity=2):
        p_total_size = 0
        q_total_size = 0
        for tensor, _, _ in high_rank_tensors:
            matrix = tensor.view(tensor.shape[0], -1)
            n, m = matrix.shape
            rank = min(n, m, self.rank)
            p_total_size += n * rank
            q_total_size += m * rank
        if self.p_memory is None:
            self.p_memory = torch.empty(p_total_size, device=self.device)
            self.q_memory = torch.empty(q_total_size, device=self.device)

        # Find them again and make lists of pointers
        ps = []
        qs = []
        p_idx = 0
        q_idx = 0
        for tensor, _, _ in high_rank_tensors:
            matrix = tensor.view(tensor.shape[0], -1)
            n, m = matrix.shape
            rank = min(n, m, self.rank)
            ps.append(self.p_memory[p_idx : p_idx + n * rank].view(n, rank))
            qs.append(self.q_memory[q_idx : q_idx + m * rank].view(m, rank))
            p_idx += n * rank
            q_idx += m * rank
        enc_end_allocate_memory.record()
        torch.cuda.synchronize()
        allocate_memory_cost = float(enc_start_allocate_memory.elapsed_time(enc_end_allocate_memory)) / 1000.0
        encode_decode_counter += allocate_memory_cost

        enc_start_prepare_q.record()
        for (tensor, _, _), q, p in zip(high_rank_tensors, qs, ps):
            matrix = tensor.view(tensor.shape[0], -1)
            n, m = matrix.shape

            if self.reuse_query and not memory_is_uninitialized:
                # orthogonalize(q)
                pass
            else:
                # Sample a query vector q
                self.set_random(q)
        enc_end_prepare_q.record()
        torch.cuda.synchronize()
        prepare_q_cost = float(enc_start_prepare_q.elapsed_time(enc_end_prepare_q)) / 1000.0
        encode_decode_counter += prepare_q_cost

        enc_start_compute_p.record()
        for (tensor, _, _), q, p in zip(high_rank_tensors, qs, ps):
            matrix = tensor.view(tensor.shape[0], -1)
            torch.matmul(matrix, q, out=p)
        enc_end_compute_p.record()
        torch.cuda.synchronize()
        compute_p_cost = float(enc_start_compute_p.elapsed_time(enc_end_compute_p)) / 1000.0
        encode_decode_counter += compute_p_cost

        comm_start_reduce_p.record()
        all_reduce(self.p_memory)
        bits_communicated += n_bits(self.p_memory)
        comm_end_reduce_p.record()
        torch.cuda.synchronize()
        reduce_p_cost = float(comm_start_reduce_p.elapsed_time(comm_end_reduce_p)) / 1000.0
        comm_time_counter += reduce_p_cost        

        # Start communicating rank 1 tensors
        enc_start_r1_pack.record()
        rank1_tensor_list = TensorBuffer([tensor for (tensor, _, _) in rank1_tensors])
        enc_end_r1_pack.record()
        torch.cuda.synchronize()
        r1_pack_cost = float(enc_start_r1_pack.elapsed_time(enc_end_r1_pack)) / 1000.0
        encode_decode_counter += r1_pack_cost

        comm_start_rank1_allreduce.record()
        #rank1_handle = rank1_tensor_list.all_reduce(async_op=False)
        rank1_tensor_list.all_reduce(async_op=False)
        bits_communicated += rank1_tensor_list.bits()
        comm_end_rank1_allreduce.record()
        torch.cuda.synchronize()
        rank1_allreduce_cost = float(comm_start_rank1_allreduce.elapsed_time(comm_end_rank1_allreduce)) / 1000.0
        comm_time_counter += rank1_allreduce_cost         

        enc_start_normalize_p.record()
        for p in ps:
            orthogonalize(p)
        enc_end_normalize_p.record()
        torch.cuda.synchronize()
        normalize_p_cost = float(enc_start_normalize_p.elapsed_time(enc_end_normalize_p)) / 1000.0
        encode_decode_counter += normalize_p_cost

        enc_start_compute_q.record()
        for p, q, (tensor, _, _) in zip(ps, qs, high_rank_tensors):
            matrix = tensor.view(tensor.shape[0], -1)
            torch.matmul(matrix.t(), p, out=q)
        enc_end_compute_q.record()
        torch.cuda.synchronize()
        compute_q_cost = float(enc_start_compute_q.elapsed_time(enc_end_compute_q)) / 1000.0
        encode_decode_counter += compute_q_cost

        comm_start_reduce_q.record()
        all_reduce(self.q_memory)
        bits_communicated += n_bits(self.q_memory)
        comm_end_reduce_q.record()
        torch.cuda.synchronize()
        reduce_q_cost = float(comm_start_reduce_q.elapsed_time(comm_end_reduce_q)) / 1000.0
        comm_time_counter += reduce_q_cost        

        dec_start_outerprod.record()
        self.q_memory.data[:] /= self.n_workers
        for p, q, (tensor, out, mem) in zip(ps, qs, high_rank_tensors):
            # Set the output gradient
            torch.matmul(p, q.t(), out=out.data[:])
            mem.data[:] = tensor - out
        dec_end_outerprod.record()
        torch.cuda.synchronize()
        outerprod_cost = float(dec_start_outerprod.elapsed_time(dec_end_outerprod)) / 1000.0
        encode_decode_counter += outerprod_cost        

        dec_start_r1_unpack.record()
        #rank1_handle.wait()
        rank1_tensor_list.buffer /= self.n_workers
        rank1_tensor_list.unpack([out for (_, out, _) in rank1_tensors])
        dec_end_r1_unpack.record()
        torch.cuda.synchronize()
        r1_unpack_cost = float(dec_start_r1_unpack.elapsed_time(dec_end_r1_unpack)) / 1000.0
        encode_decode_counter += r1_unpack_cost         

        return bits_communicated, comm_time_counter, encode_decode_counter



@torch.jit.script
def orthogonalize(matrix, eps=torch.tensor(1e-8)):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2)) + eps
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col