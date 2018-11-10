# TensorRT 5 Execution Sample

## Prerequisite

[TensorRT installation](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing)

## TensorFlow model conversion to Uff

```
$curl -O http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz
$tar zxvf mobilenet_v1_1.0_224.tgz
$convert-to-uff mobilenet_v1_1.0_224_frozen.pb -o mobilenet_v1_1.0_224.uff
```

## Run

```
$python run_tensorrt.py cat.jpg
build engine...
allocate buffers
load input
Inference time 0: 0.8375644683837891 [msec]
Inference time 1: 0.7922649383544922 [msec]
Inference time 2: 0.8244514465332031 [msec]
Inference time 3: 0.7870197296142578 [msec]
Inference time 4: 0.8032321929931641 [msec]
Inference time 5: 0.7877349853515625 [msec]
Inference time 6: 0.8432865142822266 [msec]
Inference time 7: 0.8103847503662109 [msec]
Inference time 8: 0.8285045623779297 [msec]
Inference time 9: 0.8194446563720703 [msec]

Classification Result:
1 tiger cat 0.45609813928604126
2 cougar 0.2968028783798218
3 Persian cat 0.24455446004867554
4 leopard 0.00040596097824163735
5 cheetah 0.00032873026793822646
```
