# Inference Benchmark: Jetson AGX Orin

## Latency Comparison (PointPillars)

| Configuration | Latency (ms) | FPS | Speedup |
|--------------|-------------|-----|---------|
| PyTorch FP32 (baseline) | 91.2 | 11.0 | 1.0x |
| PyTorch FP16 | 58.4 | 17.1 | 1.56x |
| TensorRT FP32 | 41.3 | 24.2 | 2.21x |
| TensorRT FP16 | **23.8** | **42.0** | **3.83x** |
| TensorRT INT8 | 18.1 | 55.2 | 5.03x (accuracy loss) |

Selected: TensorRT FP16 - best latency/accuracy tradeoff

## Memory Usage (Jetson AGX Orin)
| Config | GPU Memory (MB) | CPU Memory (MB) |
|--------|----------------|----------------|
| PyTorch FP32 | 2,841 | 312 |
| TensorRT FP16 | 1,203 | 187 |
| TensorRT INT8 | 891 | 174 |

## Tracking Performance (ByteTrack 3D)
| Metric | Value |
|--------|-------|
| MOTA | **78.4%** |
| MOTP | 0.82 |
| IDs switched | 23 |
| FP rate | 8.2% |
| FN rate | 13.4% |
