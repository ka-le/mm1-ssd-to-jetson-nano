# mm1-ssd-to-jetson-nano

# baseline FP32
trtexec --onnx=models/mb1-ssd.onnx --saveEngine=models/mb1-ssd_fp32.engine \
    --explicitBatch

# FP16 (big win on Jetson's Tensor Cores)
trtexec --onnx=models/mb1-ssd.onnx --saveEngine=models/mb1-ssd_fp16.engine \
    --fp16 --explicitBatch

# larger workspace (lets TensorRT consider more kernel/tactic options)
trtexec --onnx=models/mb1-ssd.onnx --saveEngine=models/mb1-ssd_fp16_ws4096.engine \
    --fp16 --workspace=4096 --explicitBatch

--workspace: megabytes of memory (e.g., 3000 MB) for TensorRT layer optimization
--fp16: Configures the Jetson Nano to run the model in 16-bit precision, doubling inference speed compared to 32-bit (FP32).

To run the engine: 
python object_detect.py models/input.jpg --output models/output.jpg

# View the engine graph
with gfile.FastGFile("trt_graph.pb", 'wb') as f:
        f.write(your_trt_graph.SerializeToString())
print("TRT model is stored!")

or try using /trt-engine-explorer/

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
