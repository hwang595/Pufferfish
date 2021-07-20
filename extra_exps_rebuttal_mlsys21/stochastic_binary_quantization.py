# implement the binary quantization proposed in
# https://arxiv.org/pdf/1611.00429.pdf
import numpy as np
import torch
import time

from compressor import compressor_modified

def n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()


def quantization_mapper(grad_vec, device="cpu"):
    # quantization described in section 2.1
    #quantization_start = time.time()
    quantization_start = torch.cuda.Event(enable_timing=True)
    quantization_end = torch.cuda.Event(enable_timing=True)

    quantization_start.record()
    #torch.cuda.synchronize()
    #quantization_start = time.time()
    min_val = torch.min(grad_vec)
    max_val = torch.max(grad_vec)
    prob_vector = (grad_vec - min_val)/(max_val-min_val)
    dices_vector = torch.bernoulli(prob_vector) # binary
    dices_vector = dices_vector * 2 - 1
    quantization_end.record()
    torch.cuda.synchronize()
    #quantization_end = time.time()
    #quantization_cands = [max_val, min_val]
    #quantized_grad_vec = np.random.choice(quantization_cands, grad_vec.shape[0], p=prob_vector)
    #encode_start = time.time()

    encode_start = torch.cuda.Event(enable_timing=True)
    encode_end = torch.cuda.Event(enable_timing=True)
    #torch.cuda.synchronize()
    encode_start.record()
    #encode_start = time.time()
    compressor = compressor_modified(using_cuda = True, local_rank = 0)
    compressed_dice, dice_size = compressor.compress(dices_vector)
    num_bits_comm = n_bits(compressed_dice)
    print("n bits comm: {}".format(num_bits_comm))
    encode_end.record()
    torch.cuda.synchronize()
    #encode_end = time.time()


    decode_start = torch.cuda.Event(enable_timing=True)
    decode_end = torch.cuda.Event(enable_timing=True) 
    decode_start.record()
    recovered_dice = compressor.uncompress(compressed_dice, dice_size)
    recovered_dice = (recovered_dice + 1) / 2
    quantized_grad_vec = torch.where(recovered_dice==1, max_val, min_val)
    print("dices_vector: {}, recovered_dice: {}".format(dices_vector, recovered_dice))
    decode_end.record()
    torch.cuda.synchronize()

    #print("##### quant cost: {}, encode cost: {}".format(quantization_end-quantization_start, encode_end-encode_start))
    print("##### quant cost: {}, encode cost: {}, decode cost: {}".format(quantization_start.elapsed_time(quantization_end), 
                                                            encode_start.elapsed_time(encode_end), decode_start.elapsed_time(decode_end)))

    #print("dices vector: {}, recovered_dice: {}".format(dices_vector, recovered_dice))
    #quantized_grad_vec = torch.where(dices_vector==1, max_val, min_val)
    #print("max:{}, min:{}, quantized:{}".format(max_val, min_val, quantized_grad_vec))
    return quantized_grad_vec


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_vec = torch.randn(int(25610205)).to(device)
    n_bits_raw = n_bits(grad_vec)
    print("n bits raw: {}".format(n_bits_raw))
    #grad_vec = torch.randn(int(10000000)).to(device)
    quantized_grad_vec = quantization_mapper(grad_vec=grad_vec, device=device)
    #print(quantized_grad_vec)