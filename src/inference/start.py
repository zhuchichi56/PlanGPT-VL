import os
import subprocess
import argparse
import signal
import time
import sys
from loguru import logger

def start_single_gpu_servers(model_path: str, base_port: int, gpu_ids: list,
                           gpu_memory_utilization: float = 0.85):
    """启动多个单GPU服务器进程"""
    processes = []

    for i, gpu_id in enumerate(gpu_ids):
        port = base_port + i

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            "python", "server.py",
            "--port", str(port),
            "--model_path", model_path,
            "--tensor_parallel_size", "1",
            "--gpu_memory_utilization", str(gpu_memory_utilization)
        ]

        logger.info(f"Starting single-GPU server on port {port} using GPU {gpu_id}")
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(cmd, env=env)
            processes.append((process, port, gpu_id))
            logger.info(f"Server started with PID {process.pid}")
            time.sleep(5)  # 给每个服务器一些启动时间
        except Exception as e:
            logger.error(f"Failed to start server on port {port}: {e}")
            # 清理已启动的进程
            stop_servers(processes)
            return None

    return processes

def start_multi_gpu_servers(model_path: str, base_port: int, gpu_ids: list,
                           tensor_parallel_size: int, gpu_memory_utilization: float = 0.85):
    """启动多个多GPU张量并行服务器进程"""
    num_servers = len(gpu_ids) // tensor_parallel_size
    if len(gpu_ids) % tensor_parallel_size != 0:
        logger.warning(f"GPU数量{len(gpu_ids)}不能被tensor_parallel_size{tensor_parallel_size}整除,最后一个服务器会使用剩余的GPU")
        num_servers += 1

    processes = []

    for server_idx in range(num_servers):
        start_gpu_idx = server_idx * tensor_parallel_size
        end_gpu_idx = min(start_gpu_idx + tensor_parallel_size, len(gpu_ids))
        server_gpu_ids = gpu_ids[start_gpu_idx:end_gpu_idx]
        server_tp_size = len(server_gpu_ids)
        port = base_port + server_idx

        env = os.environ.copy()
        # 设置物理GPU映射
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, server_gpu_ids))
        logger.info(f"Server {server_idx+1}: Using physical GPUs {server_gpu_ids}")

        cmd = [
            "python", "server.py",
            "--port", str(port),
            "--model_path", model_path,
            "--tensor_parallel_size", str(server_tp_size),
            "--gpu_memory_utilization", str(gpu_memory_utilization)
        ]

        logger.info(f"Starting multi-GPU server {server_idx+1}/{num_servers} on port {port} using GPUs {server_gpu_ids}")
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(cmd, env=env)
            processes.append((process, port, server_gpu_ids))
            logger.info(f"Server {server_idx+1} started with PID {process.pid}")
            time.sleep(10)  # 多GPU服务器需要更多启动时间
        except Exception as e:
            logger.error(f"Failed to start server {server_idx+1} on port {port}: {e}")
            # 清理已启动的进程
            stop_servers(processes)
            return None

    return processes

def stop_servers(processes):
    """停止所有服务器进程"""
    if not processes:
        return

    logger.info("Stopping all servers...")
    for process, port, gpu_info in processes:
        if process.poll() is None:
            logger.info(f"Stopping server on port {port} (GPU {gpu_info}, PID {process.pid})")
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    logger.info("All servers stopped.")

def start_servers(model_path: str, base_port: int = 8000, tensor_parallel_size: int = 1,
                 gpu_ids: str = "0", gpu_memory_utilization: float = 0.85):
    """启动服务器的统一入口函数"""
    gpu_list = [int(gpu.strip()) for gpu in gpu_ids.split(",")]

    # 设置环境变量
    os.environ["SERVER_URL"] = f"http://localhost:{base_port}"
    os.environ["GPU_IDS"] = gpu_ids

    if tensor_parallel_size == 1:
        # 单GPU模式: 每个GPU启动一个进程
        logger.info(f"Starting {len(gpu_list)} single-GPU servers on ports {base_port}-{base_port + len(gpu_list) - 1}")
        processes = start_single_gpu_servers(model_path, base_port, gpu_list, gpu_memory_utilization)
    else:
        # 多GPU张量并行模式: 每tensor_parallel_size个GPU启动一个进程
        logger.info(f"Starting servers with tensor_parallel_size={tensor_parallel_size}")
        processes = start_multi_gpu_servers(model_path, base_port, gpu_list, tensor_parallel_size, gpu_memory_utilization)

    if processes is None:
        logger.error("Failed to start servers")
        return False

    try:
        # 打印服务器信息
        logger.info("=" * 50)
        logger.info("所有服务器已启动:")
        for process, port, gpu_info in processes:
            server_url = f"http://localhost:{port}"
            logger.info(f"  端口 {port}: GPU {gpu_info} -> {server_url}")
        logger.info("=" * 50)
        logger.info("按 Ctrl+C 停止所有服务器")

        # 保持运行直到收到中断信号
        signal.pause()

    except KeyboardInterrupt:
        logger.info("收到中断信号,正在关闭服务器...")
    finally:
        stop_servers(processes)
        logger.info("程序退出")

    return True

def main():
    """Main function"""
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>")

    parser = argparse.ArgumentParser(description="启动VLM推理服务器")
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型路径")
    parser.add_argument("--port", type=int, default=8000,
                       help="基础端口 (default: 8000)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="张量并行GPU数量 (default: 1, 设置为2会启动2个进程)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                       help="GPU内存利用率 (0.0-1.0, default: 0.85)")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3",
                       help="使用的GPU ID列表,逗号分隔 (default: 0,1,2,3)")

    args = parser.parse_args()

    # 解析GPU列表
    gpu_list = [int(gpu.strip()) for gpu in args.gpu_ids.split(",")]

    logger.info("=" * 60)
    logger.info("VLM推理服务器启动配置")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"基础端口: {args.port}")
    logger.info(f"张量并行大小: {args.tensor_parallel_size}")
    logger.info(f"GPU列表: {gpu_list}")
    logger.info(f"GPU内存利用率: {args.gpu_memory_utilization}")

    if args.tensor_parallel_size == 1:
        logger.info(f"将启动 {len(gpu_list)} 个单GPU服务器进程")
        logger.info(f"端口范围: {args.port}-{args.port + len(gpu_list) - 1}")
    else:
        num_servers = len(gpu_list) // args.tensor_parallel_size
        if len(gpu_list) % args.tensor_parallel_size != 0:
            num_servers += 1
        logger.info(f"将启动 {num_servers} 个多GPU服务器进程")
        logger.info(f"每个服务器使用 {args.tensor_parallel_size} 个GPU")
        logger.info(f"端口范围: {args.port}-{args.port + num_servers - 1}")

    logger.info("=" * 60)

    # 启动服务器
    success = start_servers(
        args.model_path,
        args.port,
        args.tensor_parallel_size,
        args.gpu_ids,
        args.gpu_memory_utilization
    )

    if not success:
        logger.error("服务器启动失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
    
    
    
# python start.py --model_path /volume/pt-train/models/Qwen2.5-VL-32B-Instruct --tensor_parallel_size 1 --gpu_ids "0,1,2,3,4,5,6,7"
