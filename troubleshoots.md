# TensorRT Version Troubleshooting Guide

## Issue: AttributeError - 'max_workspace_size' not found

### Error Message
```
AttributeError: 'tensorrt.tensorrt.Builder' object has no attribute 'max_workspace_size'
```

### Cause
This error occurs due to API changes across different TensorRT versions. The method for setting workspace size has evolved:
- In older versions (7.x and earlier), it was `builder.max_workspace_size`
- In TensorRT 8.x, it moved to `builder_config.max_workspace_size`
- In TensorRT 11.0, it has been replaced by `builder_config.set_memory_pool_limit()`

### Solution

#### For TensorRT 11.0
Use `set_memory_pool_limit()` with the appropriate memory pool type:
```python
import tensorrt as trt

builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256MiB
```

#### For TensorRT 8.x
Use `max_workspace_size` as part of the builder config:
```python
import tensorrt as trt

builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
config = builder.create_builder_config()
config.max_workspace_size = 1 << 28  # 256MiB
```

#### For TensorRT 7.x and earlier
The attribute was directly on the builder:
```python
import tensorrt as trt

builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
builder.max_workspace_size = 1 << 28  # 256MiB
```

### Version Compatibility Table

| TensorRT Version | Method to Set Workspace                    | Example Code |
|------------------|-------------------------------------------|--------------|
| 11.0 and later   | `builder_config.set_memory_pool_limit()`  | `config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)` |
| 8.x - 10.x       | `builder_config.max_workspace_size`        | `config.max_workspace_size = 1 << 28` |
| 7.x and earlier  | `builder.max_workspace_size`               | `builder.max_workspace_size = 1 << 28` |

### Additional Tips
1. Always check the TensorRT release notes when upgrading versions
2. Be aware of API changes between major versions
3. Use version-specific documentation
4. Test thoroughly when upgrading TensorRT versions

### Common Pitfalls
- Using outdated code examples from the internet
- Not updating all relevant code when upgrading TensorRT
- Assuming API compatibility between major versions

## Related Resources
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [TensorRT Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)