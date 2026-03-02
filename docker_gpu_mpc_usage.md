# EzPC GPU-MPC 容器启动说明

本说明用于在 `he_compiler/EzPC` 目录下使用持久化 Docker 容器运行 `GPU-MPC`。

## 1. 进入目录

```bash
cd /home/hedong/project/he_compiler/EzPC
```

## 2. 首次使用（只需执行一次）

```bash
chmod +x docker_gpu_mpc.sh
./docker_gpu_mpc.sh build
./docker_gpu_mpc.sh create
```

说明：
- `build`：根据 `GPU-MPC/Dockerfile_Gen` 构建镜像 `ezpc-gpu-mpc:cuda11.8`
- `create`：创建持久容器 `ezpc-gpu-mpc`，并挂载宿主机 `EzPC` 目录到容器 `/workspace/EzPC`

## 3. 每次使用时进入容器

```bash
./docker_gpu_mpc.sh enter
```

进入后默认工作目录为：

```bash
/workspace/EzPC/GPU-MPC
```

## 4. 常用容器管理命令

```bash
./docker_gpu_mpc.sh status    # 查看容器状态
./docker_gpu_mpc.sh start     # 启动容器
./docker_gpu_mpc.sh stop      # 停止容器
./docker_gpu_mpc.sh restart   # 重启容器
./docker_gpu_mpc.sh logs      # 查看容器日志
```

## 5. 备注

- 容器为持久容器，不会在退出 shell 后自动删除。
- 代码在宿主机，容器内通过挂载实时访问，不会丢失修改。
- 若你的 Docker 需要 sudo，请在命令前加 `sudo`，例如：

```bash
sudo ./docker_gpu_mpc.sh enter
```
