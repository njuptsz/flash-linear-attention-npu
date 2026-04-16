# Mock HCCL Server Framework

## 概览

本框架提供了一个**通用的 HCCL Server 端模拟实现**，用于在单卡环境下替代真实的 CCU Server / HCCL 通信后端。它与具体的 MC2 算子无关 — 任何通过 workspace 消息协议与 HCCL Server 交互的算子（如 AllGatherMatmul、MatmulReduceScatter 等）都可以使用本框架进行单卡测试。

### 背景：hcclClient / Server 架构

在真实的多卡环境中，MC2 类算子的通信流程如下：

```
Device (kernel)                      CCU (HCCL Server)
  │                                    │
  │  1. 填 sendMsgs[slot]              │
  │  2. 写 commitTurnCnt[slot]         │
  │  ──────── workspace ──────────→    │
  │                                    │  3. 读 commitTurnCnt, 发现有效
  │                                    │  4. 读 sendMsgs, 执行集合通信
  │                                    │  5. 写 finishedTurnCnt[slot]
  │  ←──────── workspace ─────────     │
  │  6. 轮询 finishedTurnCnt, 继续     │
  │                                    │
```

kernel 通过 workspace 中的消息协议向 Server 提交通信请求，Server 执行实际的跨 rank 数据搬移，完成后通过 workspace 回写完成标志。这个 **workspace 消息协议是算子无关的** — 不同的 MC2 算子（AllGather、ReduceScatter 等）使用相同的协议格式，只是 commType 字段不同。

**本框架就是在 host 侧用一个轮询线程模拟这个 Server 的行为。**

```
目录结构:
tests/mc2_mock_test_frame/              # 本目录 — 通用 mock 框架
├── mock_framework.h                    # C++ 核心 (MockContextBuilder + MultiRankMockContext + MockHcclServer)
├── mock_framework_test.cpp             # C++ 单元测试 (协议交互验证)
├── mock_framework.cpp                   # pybind11 torch extension (Python binding)
└── README.md                           # 本文档

tests/torch_extension_tests/mc2/all_gather_matmul_v3/  # V3 算子测试
├── conftest.py                             # mock 测试公共工具
├── test_all_gather_matmul_v3_mock.py       # 单 rank mock 测试
├── test_all_gather_matmul_v3_mock_multirank.py  # 多 rank 多流 mock 测试
└── test_all_gather_matmul_v3.py            # 真实多卡测试 (torchrun)
```

---

## 1. Workspace 消息协议

MockHcclServer 模拟的核心是 workspace 消息协议。这是 HCCL 定义的 kernel ↔ Server 通信接口，与具体算子无关。

### 1.1 Workspace 布局

workspace 需要 512B 对齐，包含 4 个区域：

```
Workspace (512B 对齐):
┌──────────────────────────────────────┐
│ sendMsgs[64]      (each 112B)        │ +0x0000   kernel → server 的通信请求
├──────────────────────────────────────┤
│ recvMsgs[64]      (each 112B)        │ +0x1C00   server → kernel 的响应 (预留)
├──────────────────────────────────────┤
│ commitTurnCnt[64] (each 64B)         │ +0x5800   kernel 提交标志
├──────────────────────────────────────┤
│ finishedTurnCnt[64](each 64B)        │ +0x6800   server 完成标志
└──────────────────────────────────────┘
```

### 1.2 消息结构

**HcclMsg (112 bytes)** — kernel 填写的通信请求：

| 偏移 | 字段 | 说明 |
|------|------|------|
| +0x00 | commType (u32) | 通信类型: AllGather=6, ReduceScatter=7 |
| +0x04 | opType (u32) | reduce 操作类型 |
| +0x08 | sendBuffer (u64) | device 源地址 |
| +0x10 | recvBuffer (u64) | device 目标地址 |
| +0x18 | dataCnt (u64) | 元素数 |
| +0x20 | strideCount (u64) | recvBuf 中 rank 间步长 |
| +0x28 | msgValid (u32) | HCCL_MSG_VALID_MASK (0x5CDF123A) |
| +0x2C | hcclDataType (u32) | 数据类型: FP32=0, FP16=1, BF16=5, ... |
| +0x30 | rest[64] | 剩余字段 |

**TurnCnt (64 bytes)** — 提交/完成计数器：

| 偏移 | 字段 | 说明 |
|------|------|------|
| +0x00 | valid (u64) | COMMIT_VALID_MASK (987654321) 表示有效 |
| +0x08 | cnt (u64) | 提交/完成计数 |
| +0x10 | reserved[6] | 填充至 cache line 对齐 |

### 1.3 协议流程

**常规通信（AllGather/ReduceScatter）：**

```
kernel:
  1. 填写 sendMsgs[slot] (commType, sendBuf, recvBuf, dataCnt, ...)
  2. 写 commitTurnCnt[slot].valid = COMMIT_VALID_MASK

server (MockHcclServer):
  3. 轮询 commitTurnCnt, 发现 valid → 读 sendMsgs[slot]
  4. 根据 commType 执行操作 (单卡: D2D memcpy sendBuf → recvBuf)
  5. 写 finishedTurnCnt[slot].cnt = commitCnt
  6. 清除 commitTurnCnt[slot].valid

kernel:
  7. 轮询 finishedTurnCnt[slot].cnt >= 期望值 → 继续
```

**Finalize（通信结束握手）：**

```
kernel Finalize():
  1. 提交 finalize commit (最后一条消息)

server:
  2. 检测到 finalize commit
  3. 写 finishedTurnCnt[slot].cnt = FINALIZE_FINISH_CNT (1234567899999999999)

kernel:
  4. 检测到 FINALIZE_FINISH_CNT → 通信流程结束
```

---

## 2. 框架组件

### 2.1 MockHcclServer — 通用 HCCL Server 模拟

**核心职责：** 在 host 端启动一个轮询线程，监听 workspace 中的通信请求，并使用传入的各 rank 输入 tensor 模拟真实的集合通信语义。

构造时需传入每个 rank 的输入 tensor（device 指针），server 在处理通信请求时读取这些 tensor，模拟从远端 rank 获取数据。

**支持的通信类型：**

| commType | 操作 | Mock 行为 |
|----------|------|-----------|
| 6 (AllGather) | 各 rank chunk 拼接 | rankInputs[r] → recvBuf[r * stride]，本卡用 sendBuf |
| 7 (ReduceScatter) | reduce + scatter | D2H 读各 rank → host 端 element-wise reduce → 取 chunk[localRank] H2D 写回 |
| 2 (AllReduce) | reduce 全量 | D2H 读各 rank → host 端 element-wise reduce → 全量 H2D 写回 |
| 12 (AlltoAll) | 块重排 | 各 rank 的 block[localRank] → recvBuf[r] |

**Reduce 操作：** 支持 SUM(0) / PROD(1) / MAX(2) / MIN(3)，host 侧通过 FP16/BF16 ↔ float 转换后计算。

### 2.2 MockContextBuilder — 单 rank HCCL Context 构造

在 device 侧构造 kernel 需要的 `HcclA2CombineOpParam` 结构（即 commContext）。所有 rank 的 `windowsIn[i]` 指向同一块 device 内存，使得 910B AIV 模式的 flag 同步机制自洽。

### 2.3 MultiRankMockContext — 多 rank Context 构造

分配 N 个独立 window（184MB 每个）+ N 个 context，context 中 windowsIn[] 互相交叉引用。配合多 stream 并发 kernel 实现真实多 rank 通信。

### 2.4 Python Binding (mock_framework.cpp)

通过 pybind11 暴露给 Python：

```python
import mock_hccl_ext

# 单 rank 测试
ctx = mock_hccl_ext.MockContext(rank_num=2, rank_id=0, device_id=0)
ctx.build()
server = mock_hccl_ext.MockServer(workspace_ptr=ctx.workspace_ptr(),
                                   rank_inputs=[t0, t1], local_rank_id=0)
server.start()
# ... 调用算子 ...
server.wait_for_finalize(slot=0, timeout_ms=10000)
server.stop()

# 多 rank 测试
mctx = mock_hccl_ext.MultiRankContext(rank_num=2, device_id=0)
mctx.build()
ctx_r0 = mctx.context_tensor(0)
ctx_r1 = mctx.context_tensor(1)
# Launch on separate streams...
```

---

## 3. 单元测试

`mock_framework_test.cpp` 验证 mock server 的协议正确性（与具体算子无关）：

| Test | 验证点 |
|------|--------|
| TestContextBuilder | context H2D/D2H，字段 readback 正确 |
| TestServerProtocol | 手写 commitTurnCnt → server 响应 finishedTurnCnt |
| TestServerDataCopy | AllGather msg → server 执行 D2D copy → recvBuf 内容正确 |
| TestFinalizeProtocol | regular msg + finalize commit → FINALIZE_FINISH_CNT 响应 |

编译运行：

```bash
ASCEND_HOME=~/Ascend/ascend-toolkit/latest
g++ -std=c++17 -O2 -I${ASCEND_HOME}/include \
  mock_framework_test.cpp \
  -L${ASCEND_HOME}/lib64 -lascendcl -lpthread \
  -Wl,-rpath,${ASCEND_HOME}/lib64 \
  -o /tmp/mock_framework_test
/tmp/mock_framework_test
```

---

## 4. 如何为新算子添加测试

本框架与算子无关，为新的 MC2 算子添加测试只需：

1. **构造 mock context** — `MockContextBuilder.Build(rankNum, rankId)`
2. **启动 mock server** — `MockHcclServer(workspace).Start()`
3. **调用算子** — 将 mock context 作为 commContext 输入传给算子
4. **等待完成** — `WaitForFinalize()` 等待 kernel 的 Finalize 握手
5. **验证结果** — 检查算子的计算输出是否正确

---

## 5. 已知限制

| 限制 | 说明 |
|------|------|
| 单卡 only | 多卡需要真实 HCCL 或多 device mock |
| D2D memcpy 代替集合通信 | 单卡无跨 rank 数据，AllGather/ReduceScatter 退化为 memcpy |
| Mock context 无 RDMA | IbVerbsData / aiRMAInfo 为 0，不支持跨节点场景 |
| Server 轮询延迟 | host D2H/H2D 轮询有 ~50us 级延迟，不反映真实通信时序 |
| Flag 自解不验证时序 | 单卡 mock 的 flag 写后立即可读，无法验证多卡竞争条件 |
