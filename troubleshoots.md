# TensorRT Troubleshooting Guide

## Issue 1: AttributeError - 'max_workspace_size' not found

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

<details>
<summary>Click to expand solutions for different TensorRT versions</summary>

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

</details>

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

## Issue 2: TensorRT Installation Errors

### Error Message
Various error messages may occur during the installation process, typically when running `pip install -e .`

### Cause
Installation errors can occur due to outdated package managers, missing dependencies, or version incompatibilities.

### Solution

<details>
<summary>Click to expand step-by-step solution</summary>

If you encounter TensorRT errors during installation when running `pip install -e .`, try the following steps:

1. Upgrade pip:
   ```
   python3 -m pip install --upgrade pip
   ```

2. Install wheel:
   ```
   python3 -m pip install wheel
   ```

3. Upgrade TensorRT:
   ```
   python3 -m pip install --upgrade tensorrt
   ```

These steps can help resolve compatibility issues and ensure you have the latest versions of pip, wheel, and TensorRT installed.

</details>

### Additional Tips
1. Make sure your system meets the prerequisites for TensorRT installation
2. Check for any specific installation instructions for your TensorRT version
3. Ensure your CUDA and cuDNN versions are compatible with the TensorRT version you're installing

## Related Resources
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [TensorRT Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
- [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)