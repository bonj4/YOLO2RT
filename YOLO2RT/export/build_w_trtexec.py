import subprocess
from typing import Dict, Any, Optional,List


def convert_onnx_to_engine(
        weights: str,
        fp16: bool = False,
        batch_size: int = 1,
        workspace_size: int = 4096,
        verbose: bool = False,
        builder_optimization: int = 3
) -> bool:
    try:
        command = ["/usr/src/tensorrt/bin/trtexec"]

        # Extract output path from weights path
        output_path = weights.rsplit('.', 1)[0] + '.engine'

        # Add ONNX input and engine output paths
        command.extend([
            f"--onnx={weights}",
            f"--saveEngine={output_path}"
        ])

        # Handle precision
        if fp16:
            command.append("--fp16")


        # Add batch size
        command.append(f"--maxBatch={batch_size}")

        # Add workspace size
        command.append(f"--workspace={workspace_size}MiB")

        # Add verbose logging if requested
        if verbose:
            command.append("--verbose")

        # Add builder optimization level
        command.append(f"--builderOptimizationLevel={builder_optimization}")

        # Run the command
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print(f"Successfully converted {weights} to {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Conversion failed with error:\n{e.stderr}")
        return False