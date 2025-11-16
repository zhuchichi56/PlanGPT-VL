# VLM Inference Server

多模态大语言模型推理服务器,支持单GPU和多GPU张量并行模式。

## 文件说明

- `server.py` - 服务器,支持单GPU和多GPU张量并行
- `client.py` - 客户端,包含推理功能和可直接导入的parallel_image_inference函数
- `start.py` - 服务器启动脚本,支持自动多进程启动
- `ray_inference.py` - Ray分布式推理(按需使用)

## 启动服务器

### 单GPU模式(每个GPU启动一个进程)
```bash
python start.py --model_path /path/to/your/model --tensor_parallel_size 1 --gpu_ids "0,1,2,3"
```
*这会在端口8000-8003启动4个单GPU服务器进程*

### 多GPU张量并行模式(每2个GPU启动一个进程)
```bash
python start.py --model_path /path/to/your/model --tensor_parallel_size 2 --gpu_ids "0,1,2,3"
```
*这会在端口8000-8001启动2个多GPU服务器进程*

### 大模型模式(8个GPU张量并行)
```bash
python start.py --model_path /path/to/your/model --tensor_parallel_size 8 --gpu_ids "0,1,2,3,4,5,6,7"
```
*这会在端口8000启动1个8GPU服务器进程*

## 客户端使用

### 1. 直接导入使用
```python
from client import parallel_image_inference

prompts = ["描述这张图片", "这张图片有什么特别之处?"]
image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]

# 自动分布式推理
results = parallel_image_inference(prompts, image_paths)
print(results)
```

### 2. 命令行测试
```bash
# 基础测试
python client.py --url http://localhost:8000 --test_mode health

# 图像推理测试
python client.py --url http://localhost:8000 --test_mode images --samples 3

# 指定图像测试
python client.py --url http://localhost:8000 --test_mode images --image /path/to/image.jpg --prompt "描述这张图片"
```

### 3. 测试导入功能
```bash
python test_import.py
```

## 主要特性

- **自动多进程**: 根据tensor_parallel_size自动启动相应数量的服务器进程
- **分布式推理**: parallel_image_inference函数自动分配请求到多个服务器
- **简单导入**: 可以直接`from client import parallel_image_inference`使用
- **灵活配置**: 支持单GPU、多GPU张量并行等多种模式
- **自动端口分配**: 自动分配端口范围(8000, 8001, 8002...)

## 启动逻辑

- `tensor_parallel_size=1`: 每个GPU启动一个单GPU服务器进程
- `tensor_parallel_size=2`: 每2个GPU启动一个多GPU服务器进程
- `tensor_parallel_size=4`: 每4个GPU启动一个多GPU服务器进程
- 端口从8000开始自动递增分配