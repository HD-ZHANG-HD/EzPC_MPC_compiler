# ResNet20 MPC 运行说明（容器内）

本文档整理 `tests/resnet20.cpp` 在容器内的“一键构建 + 运行”命令。

## 1. 进入容器并切换目录

```bash
cd /home/hedong/project/he_compiler/EzPC
./docker_gpu_mpc.sh enter

cd /workspace/EzPC/GPU-MPC
```

## 2. 一键构建

```bash
cmake -S tests -B tests/build-resnet20 -DCMAKE_BUILD_TYPE=Release \
  && cmake --build tests/build-resnet20 -j4 --target resnet20
```

可执行文件路径：

```bash
./tests/build-resnet20/resnet20
```

## 3. 一键本地 two-process simulation（推荐）

该命令会自动完成：
- party0/party1 的 keygen
- 启动两个评估进程进行本地 MPC 推理
- 打印每方 `iter` 和 `avg_e2e_time_ms`

```bash
GPU_MEMPOOL_WARMUP_GB=2 \
./tests/build-resnet20/resnet20 local-sim /tmp/resnet20_keys 1 12345 1
```

参数说明：
- `local-sim`
- `/tmp/resnet20_keys`：key 存储目录
- `1`：迭代次数（iters）
- `12345`：随机种子（seed）
- `1`：batch size

## 4. 分步运行（keygen + eval）

### 4.1 生成 key（两方各执行一次）

```bash
./tests/build-resnet20/resnet20 keygen 0 /tmp/resnet20_keys 12345 1
./tests/build-resnet20/resnet20 keygen 1 /tmp/resnet20_keys 12345 1
```

### 4.2 在线评估（需要两个终端并行）

终端 A（party 0）：

```bash
GPU_MEMPOOL_WARMUP_GB=2 \
./tests/build-resnet20/resnet20 eval 0 127.0.0.1 /tmp/resnet20_keys 1 12345 1
```

终端 B（party 1）：

```bash
GPU_MEMPOOL_WARMUP_GB=2 \
./tests/build-resnet20/resnet20 eval 1 127.0.0.1 /tmp/resnet20_keys 1 12345 1
```

## 5. 局域网（LAN）双机运行

以下场景适用于两台机器在同一局域网（例如 `192.168.x.x`）：
- 机器 A：`party 0`
- 机器 B：`party 1`

建议两台机器都先完成第 1、2 步（进入容器 + 构建）。

### 5.1 两台机器各自生成 key

机器 A（party 0）：

```bash
./tests/build-resnet20/resnet20 keygen 0 /tmp/resnet20_keys 12345 1
```

机器 B（party 1）：

```bash
./tests/build-resnet20/resnet20 keygen 1 /tmp/resnet20_keys 12345 1
```

### 5.2 在线评估（并行启动）

假设：
- 机器 A 局域网 IP 是 `192.168.1.10`
- 机器 B 局域网 IP 是 `192.168.1.11`

先在机器 A 启动（会等待连接）：

```bash
GPU_MEMPOOL_WARMUP_GB=2 \
./tests/build-resnet20/resnet20 eval 0 192.168.1.11 /tmp/resnet20_keys 1 12345 1
```

再在机器 B 启动（连接到 A）：

```bash
GPU_MEMPOOL_WARMUP_GB=2 \
./tests/build-resnet20/resnet20 eval 1 192.168.1.10 /tmp/resnet20_keys 1 12345 1
```

## 6. 广域网（WAN）双机运行

跨公网运行时，核心是 **party 1 能连到 party 0 的监听端口（默认 42003）**。

### 6.1 网络要求

- `party 0` 所在机器（或网关）放通 TCP `42003`
- 若 `party 0` 在 NAT 后面，需配置端口映射到容器所在主机
- 推荐优先使用 VPN（如 Tailscale / WireGuard）打通内网，再按 LAN 方式填 VPN IP

### 6.2 命令模板

假设：
- 机器 A（party 0）公网/VPN 地址：`A_ADDR`
- 机器 B（party 1）公网/VPN 地址：`B_ADDR`

机器 A：

```bash
./tests/build-resnet20/resnet20 keygen 0 /tmp/resnet20_keys 12345 1
GPU_MEMPOOL_WARMUP_GB=2 \
./tests/build-resnet20/resnet20 eval 0 B_ADDR /tmp/resnet20_keys 1 12345 1
```

机器 B：

```bash
./tests/build-resnet20/resnet20 keygen 1 /tmp/resnet20_keys 12345 1
GPU_MEMPOOL_WARMUP_GB=2 \
./tests/build-resnet20/resnet20 eval 1 A_ADDR /tmp/resnet20_keys 1 12345 1
```

> 提示：`eval` 的第 3 个参数是“对端可达地址”；如果使用 VPN，就填 VPN 分配的地址。

## 7. 常见问题

- 显存不足（`cudaErrorMemoryAllocation`）：
  - 降低 warmup：`GPU_MEMPOOL_WARMUP_GB=1` 或 `0`
  - 降低并发任务，确保 GPU 空闲
- 端口连接失败：
  - `eval` 两个进程需几乎同时启动
  - 确认未被其他进程占用默认通信端口 `42003`
  - WAN 下检查安全组/防火墙/NAT 映射是否生效
