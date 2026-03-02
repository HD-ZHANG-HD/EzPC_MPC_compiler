# BERT-base MPC 运行说明（容器内）

本文档整理 `tests/bert_base.cpp` 在容器内的“一键构建 + 运行”命令。

## 1. 进入容器并切换目录

```bash
cd /home/hedong/project/he_compiler/EzPC
./docker_gpu_mpc.sh enter

cd /workspace/EzPC/GPU-MPC
```

## 2. 一键构建

```bash
cmake -S tests -B tests/build-resnet20 -DCMAKE_BUILD_TYPE=Release \
  && cmake --build tests/build-resnet20 -j4 --target bert_base
```

可执行文件路径：

```bash
./tests/build-resnet20/bert_base
```

## 3. 一键本地 two-process simulation（推荐）

该命令会自动完成：
- dealer 生成离线 key
- 启动 server/client 两个在线评估进程
- 打印每方 `e2e_time_ms`

```bash
./tests/build-resnet20/bert_base local-sim /tmp/bert_base_keys 12345 4
```

参数说明：
- `local-sim`
- `/tmp/bert_base_keys`：运行目录（会写入 `server.dat`、`client.dat` 等）
- `12345`：随机种子（seed）
- `4`：序列长度（seq_len）

## 4. 分步运行（dealer + eval）

### 4.1 先生成离线 key（dealer）

```bash
./tests/build-resnet20/bert_base dealer /tmp/bert_base_keys 12345 4
```

### 4.2 在线评估（需要两个终端并行）

> `eval` 的 `party` 参数：`2=SERVER`，`3=CLIENT`

终端 A（SERVER）：

```bash
./tests/build-resnet20/bert_base eval 2 127.0.0.1 /tmp/bert_base_keys 12345 4
```

终端 B（CLIENT）：

```bash
./tests/build-resnet20/bert_base eval 3 127.0.0.1 /tmp/bert_base_keys 12345 4
```

## 5. 局域网（LAN）双机运行

以下场景适用于两台机器在同一局域网：
- 机器 A：SERVER（party=2）
- 机器 B：CLIENT（party=3）

建议两台机器都先完成第 1、2 步（进入容器 + 构建）。

### 5.1 在两台机器上准备同一份离线 key

推荐做法：
1. 在机器 A 执行 dealer 生成 key：

```bash
./tests/build-resnet20/bert_base dealer /tmp/bert_base_keys 12345 4
```

2. 将机器 A 的 `/tmp/bert_base_keys` 同步到机器 B（确保 `server.dat` / `client.dat` 一致）。

### 5.2 在线评估（并行启动）

假设：
- 机器 A IP：`192.168.1.10`
- 机器 B IP：`192.168.1.11`

机器 A（SERVER）：

```bash
./tests/build-resnet20/bert_base eval 2 192.168.1.11 /tmp/bert_base_keys 12345 4
```

机器 B（CLIENT）：

```bash
./tests/build-resnet20/bert_base eval 3 192.168.1.10 /tmp/bert_base_keys 12345 4
```

## 6. 广域网（WAN）双机运行

跨公网运行时，关键是双方网络可互通（默认会使用 llama 的通信端口，如 42005/42002 相关通道）。

### 6.1 网络要求

- SERVER 所在机器放通所需 TCP 端口
- 若 SERVER 在 NAT 后，配置端口映射到容器所在主机
- 推荐优先使用 VPN（Tailscale/WireGuard）后按 LAN 方式填写 VPN IP

### 6.2 命令模板

假设：
- 机器 A（SERVER）地址：`A_ADDR`
- 机器 B（CLIENT）地址：`B_ADDR`

机器 A：

```bash
./tests/build-resnet20/bert_base eval 2 B_ADDR /tmp/bert_base_keys 12345 4
```

机器 B：

```bash
./tests/build-resnet20/bert_base eval 3 A_ADDR /tmp/bert_base_keys 12345 4
```

> 提示：`eval` 的第 3 个参数是“对端可达地址”；如果使用 VPN，请填 VPN 地址。

## 7. 常见问题

- 卡在连接阶段：
  - 两端 `eval` 需几乎同时启动
  - 检查防火墙/安全组/NAT 端口映射
- 结果不一致或启动失败：
  - 确保两端使用同一份 dealer 生成的 key 文件
  - 确保两端 `seed`、`seq_len` 参数一致
- 运行时间较长：
  - BERT-base 在 MPC 下本身开销大，可先用更短 `seq_len` 做冒烟测试
