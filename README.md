# optimizations which were run
baseline:- since there is no quantization stated, it runs on fp32
trtexec --onnx-models/mb1-ssd.onnx --saveEngine=models/basecase.engine --iterations=10 --avgRuns=10

int8:
trtexec --onnx-models/mb1-ssd.onnx --saveEngine=models/basecase.engine --int8 --iterations=10 --avgRuns=10

fp16:
trtexec --onnx-models/mb1-ssd.onnx --saveEngine=models/basecase.engine --fp16 --iterations=10 --avgRuns=10

larger workspace (alloted more memory to use) to let TensorRT consider more kernel/tactic options:
trtexec --onnx-models/mb1-ssd.onnx --saveEngine=models/basecase.engine --fp16 --workspace=4096 --iterations=10 --avgRuns=10

# To run the engine: 
python3 object_detect.py models/input.jpg --output output/output.jpg --network=ssd-mobilenet-v2

# View the engine graph
trtexec --loadEngine=${NETWORK_NAME}.engine --dumpLayerInfo

| Optimization                 | What it does                                               | Typical effect                                              |
| ---------------------------- | ---------------------------------------------------------- | ----------------------------------------------------------- |
| FP16                         | Half-precision inference                                   | Faster, lower memory usage                                  |
| INT8                         | 8-bit quantization                                         | Even faster, lower memory usage, possible accuracy tradeoff |
| Builder optimization level   | Allows more exhaustive optimization during engine build    | Longer build time, potentially faster inference             |
| Workspace size / memory pool | Gives TensorRT more memory for selecting efficient tactics | Can improve performance on complex models                   |
| Timing cache                 | Reuses benchmarking results between engine builds          | Much faster engine build time                               |
| Dynamic shapes               | Supports multiple input sizes efficiently                  | Flexibility with some overhead                              |
| CUDA Graphs                  | Reduces CPU launch overhead                                | Lower latency for repeated inference                        |
| Sparsity                     | Uses sparse weights if supported                           | Faster inference on compatible GPUs and models              |

All models were checked against models/input.jpg and output/ contains the engine build, inference times and optimizaion, architecture of engine, and output image generated

# Jetson inference
in order to run this file successfully, you need to change code in the jetson-inference/utils/python/bindings CMakeList


