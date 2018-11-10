#!/usr/bin/env python3

import argparse
import numpy as np
import tensorrt as trt
import time

from PIL import Image

import pycuda.driver as cuda
import pycuda.autoinit


MAX_BATCH_SIZE = 1
MAX_WORKSPACE_SIZE = 1 << 30

LOGGER = trt.Logger(trt.Logger.WARNING)
DTYPE = trt.float32

# Model
MODEL_FILE = 'mobilenet_v1_1.0_224.uff'
INPUT_NAME = 'input'
INPUT_SHAPE = (3, 224, 224)
OUTPUT_NAME = 'MobilenetV1/Predictions/Reshape_1'

LABELS = 'class_labels.txt'

LOOP_TIMES = 10
TOP_N = 5


def allocate_buffers(engine):
    print('allocate buffers')
    
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(DTYPE))
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    return h_input, d_input, h_output, d_output


def build_engine(model_file):
    print('build engine...')

    with trt.Builder(LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = MAX_WORKSPACE_SIZE
        builder.max_batch_size = MAX_BATCH_SIZE
        if DTYPE == trt.float16:
            builder.fp16_mode = True
        parser.register_input(INPUT_NAME, INPUT_SHAPE, trt.UffInputOrder.NCHW)
        parser.register_output(OUTPUT_NAME)
        parser.parse(model_file, network, DTYPE)
        
        return builder.build_cuda_engine(network)


def load_input(img_path, host_buffer):
    print('load input')
    
    with Image.open(img_path) as img:
        c, h, w = INPUT_SHAPE
        dtype = trt.nptype(DTYPE)
        img_array = np.asarray(img.resize((w, h), Image.BILINEAR)).transpose([2, 0, 1]).astype(dtype).ravel()
        # preprocess for mobilenet
        img_array = img_array / 127.5 - 1.0
        
    np.copyto(host_buffer, img_array)


def do_inference(n, context, h_input, d_input, h_output, d_output):
    # Transfer input data to the GPU.
    cuda.memcpy_htod(d_input, h_input)

    # Run inference.
    st = time.time()
    context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])
    print('Inference time {}: {} [msec]'.format(n, (time.time() - st)*1000))

    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh(h_output, d_output)
    
    return h_output


def parse_args():
    parser = argparse.ArgumentParser(description='TensorRT execution smaple')
    parser.add_argument('img', help='input image')
    
    return parser.parse_args()


def main():
    args = parse_args()

    with open(LABELS) as f:
        labels = f.read().split('\n')
        
    with build_engine(MODEL_FILE) as engine:
        h_input, d_input, h_output, d_output = allocate_buffers(engine)
        load_input(args.img, h_input)
        
        with engine.create_execution_context() as context:
            for i in range(LOOP_TIMES):
                output = do_inference(i, context, h_input, d_input, h_output, d_output)

    pred_idx = np.argsort(output)[::-1]
    pred_prob = np.sort(output)[::-1]

    print('\nClassification Result:')
    for i in range(TOP_N):
        print('{} {} {}'.format(i + 1, labels[pred_idx[i]], pred_prob[i]))

                
if __name__ == '__main__':
    main()
