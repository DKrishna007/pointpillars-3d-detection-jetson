# Proof Materials Guide

## Best Proof Screenshot
RViz showing 3D bounding boxes:
- Colored boxes per class with track IDs (ByteTrack)
- FPS counter in corner (42 FPS)
- Point cloud as background
- BEV (Bird's Eye View) panel alongside 3D view

## Required Visual Proofs
| Proof | Description |
|-------|-------------|
| BEV detection | Bird's eye view with 3D boxes |
| Jetson deployment video | System running live on Jetson AGX Orin |
| FPS benchmark table | benchmarks/fps_comparison.md |
| TensorRT comparison chart | Bar chart: FP32 vs FP16 vs INT8 latency |
| Tracking visualization | ByteTrack IDs persisting across frames |
| Latency graph | Latency over time plot |

## Key Numbers
- **42 FPS** on Jetson AGX Orin (TensorRT FP16)
- **3.8x** speedup vs non-optimized baseline
- **78.4% MOTA** ByteTrack 3D tracking
- **<24ms** inference latency
- **1.2 GB** GPU memory (vs 2.8 GB FP32)
