# Converter
Convert torch SEA-RAFT model to onnx model.
This repo provides convert code in `convert/torch_to_onnx.py`. Usage is as follows. 

Arguments:
- `--cfg`: Configuration file path (required). Default configs are provided in `configs`(symbolic link to `code/SEA-RAFT/configs/`) or `code/SEA-RAFT/configs/`.
- `--path`: Local checkpoint file path (either --path or --url required)
- `--url`: URL to download checkpoint file (either --path or --url required) 
- `--output_onnx_file`: Output ONNX model filename (default: `raft.onnx`)
- `--opset`: ONNX opset version (default: 16)
- `--check_numerical`: Whether to verify numerical accuracy between PyTorch and ONNX models (default: True)

Usage:
```sh
python converter/torch_to_onnx.py --cfg <path/to/config> --path <path/to/checkpoint> --output_onnx_file <path/to/output/onnx>
```
or
```sh
python converter/torch_to_onnx.py --cfg <path/to/config> --url <url/to/checkpoint> --output_onnx_file <path/to/output/onnx>
```

Example:
```sh
python converter/torch_to_onnx.py --cfg configs/eval/spring-M.json --url MemorySlices/Tartan-C-T-TSKH-spring540x960-M --output_onnx_file sea-raft-spring-M-op16.onnx
```