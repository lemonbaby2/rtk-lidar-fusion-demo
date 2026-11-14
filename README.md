# RTK + LiDAR + Camera Fusion Demo

把 **RTK/INS 绝对定位**接进**激光里程计**，在 KITTI 原始数据上跑通"点云 + 视觉 + RTK 三源融合"，
并在 GPU 上把融合后的彩色点云渲染成视频。

A small, honest reference implementation: LiDAR odometry from frame-to-frame ICP, an RTK/INS
translation pose graph on top of it, a simulated RTK outage to show what the absolute constraint
is worth, and a GPU z-buffer renderer that turns the fused coloured map into an MP4.

本仓库同时是这项研究的**数据与代码存储库**，包含两版实现：

| 版本 | 位置 | 定位 |
|---|---|---|
| **v1** | 仓库根目录 | 最小可复现的融合链路：GPU ICP 里程计 + RTK 平移位姿图 + z-buffer 渲染。全部数字可查，约 1.6 k 行 |
| **v2** | [`v2/`](v2/) | 把《大地图与多楼层导航栈升级方案》六个主题完整实现并验收：建图存储、按需加载、视觉层、多楼层、自动重定位、大地图分层导航。约 7.4 k 行 |

两版共用同一份 KITTI 数据、同一套 `frame0` 约定，但 v2 不是 v1 的加壳版：前端从
point-to-point 换成 point-to-plane，优化器从手写最小二乘换成 GTSAM 因子图，新增 IMU 预积分、
地图 schema、分层规划器。**v1 保持原样不删**，因为 v1 踩的坑（坐标系统一）是 v2 能写对的前提。

---

## 目录

- [0. 三十秒看懂](#0-三十秒看懂)
- [1. 数据可用性声明](#1-数据可用性声明--data-availability)
- [2. 仓库全图](#2-仓库全图)
- [3. 版本一（根目录）：最小融合链路](#3-版本一根目录最小融合链路)
- [4. 版本二（v2）：六主题大地图导航栈](#4-版本二v2六主题大地图导航栈)
- [5. 知识点索引](#5-知识点索引)
- [6. 复现](#6-复现)
- [7. 环境与实测状态](#7-环境与实测状态)
- [8. 已知局限](#8-已知局限)
- [9. 引用与许可](#9-引用与许可)

---

## 0. 三十秒看懂

```
KITTI raw 2011_09_26_drive_0005  (154 帧 / 15.80 s / 69.43 m)
├── Velodyne HDL-64E          点云
├── 彩色相机 02 + 03          立体，基线 0.5327 m
└── OXTS RT3003 (100 Hz)      RTK 级 GNSS/INS，当作 RTK 观测量

v1:  点云 ──► 逐帧 ICP（point-to-point）──► 平移位姿图（里程计相对 + RTK 绝对）──► 重建地图 ──► GPU z-buffer ──► MP4

v2:  点云 ──► point-to-plane ICP ──┐
     IMU  ──► 预积分 ──────────────┼──► GTSAM 因子图（Huber on RTK）──► 位姿 + 速度 + bias
     RTK  ──► 平移先验 ────────────┘                                        │
                                                                            ├──► 地图存储（原子写 / journal / 分块 / LOD）
                                                                            ├──► 多楼层烘焙与按层加载
                                                                            ├──► Scan Context 重定位（L1 检索 + L2 验证）
                                                                            ├──► 全局轻量层 + 滚动窗口细层 + 分层 A*
                                                                            └──► 视觉层（动态掩膜 / 语义代价 / 立体深度 / 描述子）
```

**v1 一句话结论**：RTK 中断 40–90 帧时，纯 ICP 里程计 ATE 均值 **0.611 m**，融合后 **0.075 m**
（改善 **87.7%**），末帧误差从 1.196 m 降到 **0.0005 m**。

**v2 一句话结论**：**IMU 不是"加了就更好"**——RTK 单独丢星时它反而让误差变大（0.192 → 0.195 m），
只有 LiDAR 与 RTK **同时**退化时才体现价值（**0.952 → 0.069 m，13.8×**）。

---

## 1. 数据可用性声明 / Data Availability

| 你在找什么 | 去哪里 |
|---|---|
| **第二版实验（六主题完整栈）** | [`v2/`](v2/) |
| **全部定量结果**（JSON，数字的唯一来源） | [`v2/results/`](v2/results/) |
| **数据清单：每个文件对应文中哪张表** | [`v2/DATA.md`](v2/DATA.md) |
| **自包含可视化查看器**（双击即开，无需服务器） | [`v2/web/index.html`](v2/web/index.html) |
| **演示视频**（BEV + 前视相机） | [`v2/media/navstack_v2.mp4`](v2/media/navstack_v2.mp4) |
| **失败与教训记录**（13 条，含被证伪的假设） | [`v2/docs/RESULTS.md`](v2/docs/RESULTS.md) |
| **v1 的测量方法与调参记录** | [`docs/RESULTS.md`](docs/RESULTS.md) |

原始数据为 **KITTI raw `2011_09_26_drive_0005`**（CC BY-NC-SA 3.0，仅限非商业用途）。
该许可不允许再分发原始传感器数据，因此仓库**不包含**原始点云与图像；用
`v2/scripts/prep.py` 从 KITTI 官方站点下载后可完整重建（细节见 [`v2/DATA.md`](v2/DATA.md) 第 5 节）。

代码许可：MIT。引用方式见 [第 9 节](#9-引用与许可)。

---

## 2. 仓库全图

```
rtk-lidar-fusion-demo/
├── README.md                  本文件：版本关系、逐文件说明、知识点索引
├── LICENSE                    MIT
├── .gitignore                 忽略 raw/ data/ out/ *.npy *.zip，但保留已发布的 media/ results/ 产物
│
├── fetch/                     ── v1：取数据 ─────────────────────────────
│   ├── fetch_kitti.sh         按官方清单下载 sync / extract / calib，支持断点续传
│   └── parallel_fetch.py      HTTP Range 并行下载器（149 行）
├── prep/
│   └── prep_kitti.py          KITTI raw → 紧凑 npz 包，含标定与 IMU 解析（183 行）
├── fuse/
│   ├── fuse.py                v1 主程序：ICP 里程计 + 平移位姿图 + 融合位姿（268 行）
│   └── icp_tune.py            ICP 配准点云策略的 5 组对照（84 行）
├── render/
│   └── render_video.py        GPU z-buffer 光栅化 → MP4（429 行）
├── docs/
│   └── RESULTS.md             v1 测量方法、坐标系对齐那次错误、ICP 选型、渲染细节
├── results/                   v1 的三份结果
│   ├── summary.json           主结果：帧数 / 点数 / ATE / 融合改善 / 设备 / 耗时
│   ├── render_stats.json      渲染：154 帧 / 20 fps / 1280×720 / 5.5 s / 19.03 MB
│   └── gpu_stress.json        压力测试：2005 帧 × 305 万点 / 2.42 亿点/秒 / 89.6% 利用率
├── media/
│   ├── fusion_demo.mp4        v1 成片（19 MB）
│   └── poster.png             封面帧
│
└── v2/                        ── v2：六主题导航栈 ────────────────────────
    ├── README.md              v2 说明：六主题验收数字、第六主题排障全过程
    ├── DATA.md                数据清单：每个 JSON 对应文中哪张表、许可、复现、引用
    ├── requirements.txt       numpy / torch(cuda) / gtsam / opencv / scipy
    ├── navstack/              库代码，一个主题一个模块（13 个文件）
    ├── scripts/               可执行入口，一阶段一个 + 六个诊断脚本（18 个文件）
    ├── results/               14 份 JSON，v2 全部数字的唯一来源
    ├── docs/RESULTS.md        方法 + 结果 + 13 条失败与教训
    ├── web/                   自包含 WebGL 查看器（数据内嵌）
    └── media/                 视频与抽帧
```

**为什么根目录和 `v2/` 各自有一份 `results/`**：两版数字口径不同（v1 是 point-to-point ICP +
最小二乘平移图；v2 是 point-to-plane + GTSAM 因子图 + IMU），混在一起会让人误以为同一套代码
跑出两组数。分开后，**每个版本的数字只依赖自己那份 JSON**。

---

## 3. 版本一（根目录）：最小融合链路

### 3.1 这个版本是什么 / 不是什么

**是**

- 一条**可复现**的最小融合链路：KITTI raw → 逐帧彩色点云 → GPU ICP 里程计 → 平移位姿图
  （里程计相对约束 + RTK 绝对约束）→ 融合位姿重建全局图 → GPU 渲染 MP4。
- 一个**诚实的对照实验**：故意在第 40–90 帧切断 RTK，看纯 ICP 里程计会漂多少、融合后还剩多少。
- 全部数字都能从 `results/` 里的 JSON 复查，视频里每一帧的误差都是当帧真实值。

**不是**

- 不是完整的 LIO/SLAM 系统。没有 IMU 预积分、没有回环检测、没有因子图优化库（gtsam/ceres），
  位姿图是一个 3 自由度（平移）稀疏最小二乘。
- 没有真机 RTK 硬件。KITTI 的 OXTS RT3003 是 RTK 级 GNSS/INS 组合导航，提供**厘米级参考位姿**，
  我们把它当作 RTK 观测量来用；第 40–90 帧的"RTK 中断"是**人为模拟**的。
- 旋转部分没有做图优化：重建地图时的朝向直接取 RTK/INS 的相对姿态。

### 3.2 流水线

```
Velodyne .bin ──┐
相机 02/03 png ─┼─► prep_kitti.py ─► pts.npy / rgb.npy / pose_rtk.npy / imu_*.npy / meta.json
oxts .txt ──────┘                          │
                                           ▼
                              fuse.py: 逐帧 ICP（常速度初值，不用 RTK）
                                           │ 相对位姿链
                                           ▼
                              平移位姿图（里程计 8 : RTK 1，lstsq 求解）
                                           │ 融合位姿
                            ┌──────────────┴──────────────┐
                            ▼                             ▼
                  重建全局彩色点云              逐帧 ATE 对照（有无 RTK）
                            │
                            ▼
              render_video.py: GPU z-buffer ─► fusion_demo.mp4
```

### 3.3 逐文件说明

| 文件 | 行数 | 职责 | 涉及的知识点 |
|---|---|---|---|
| `fetch/fetch_kitti.sh` | – | 按官方清单下载 sync / extract / calib 三套 zip，支持断点续传 | KITTI 目录结构（`image_02` / `image_03` / `velodyne_points` / `oxts`） |
| `fetch/parallel_fetch.py` | 149 | 多连接 HTTP Range 下载器。高延迟链路下单连接跑不满带宽，按字节区间切片并发取，再按序拼接 | HTTP `Range` 请求、分片并行、失败重试与字节对齐 |
| `prep/prep_kitti.py` | 183 | 把 raw 目录解析成 `pts.npy` / `rgb.npy` / `pose_rtk.npy` / IMU 三件套 / `meta.json` | 标定链 `Tr_velo→cam`、Mercator 投影转局部平面、`frame0` 归一化、固定种子子采样 |
| `fuse/fuse.py` | 268 | v1 主程序：ICP 里程计 → 平移位姿图（`lstsq`）→ 融合位姿 → 重建地图 | point-to-point ICP、常速度初值、稀疏最小二乘、RTK 中断注入、坐标系对齐 |
| `fuse/icp_tune.py` | 84 | 5 组配准点云策略对照（体素 vs 随机采样 × 半径 × 点数） | 子采样的几何保真、近场/远场权衡、中位最近邻残差作为配准质量指标 |
| `render/render_video.py` | 429 | PyTorch 写 z-buffer：投影 → 像素索引 → `scatter_reduce_(amin)` 求最近深度 → 上色；2× 超采样后 splat 再 `INTER_AREA` 降采样 | z-buffer 遮挡求解（非画家算法）、`scatter_reduce` 归约、超采样与 splat 的抗锯齿取舍 |
| `docs/RESULTS.md` | – | 测量方法 + 坐标系对齐那次错误 + ICP 选型表 + 渲染与 GPU 记录 | ATE 定义、对齐用 INS 初值 vs 事后最优拟合、步长均值作为判错工具 |
| `results/summary.json` | – | 主结果。**改动算法这个文件必然跟着变**，无法悄悄美化 | 结果即契约 |
| `results/render_stats.json` | – | 渲染吞吐与 GPU 利用率 | – |
| `results/gpu_stress.json` | – | 全量点云连续光栅化的压力测试 | – |
| `media/fusion_demo.mp4` | 19 MB | 成片 | – |
| `media/poster.png` | – | 封面 | – |

### 3.4 关键结果

| 指标 | 纯 ICP 里程计 | 融合（+RTK） |
|---|---|---|
| ATE 均值 | 0.611 m | **0.075 m** |
| ATE 末帧 | 1.196 m | **0.0005 m** |
| 改善 | – | **87.7%** |

| 规模与性能 | 数值 |
|---|---|
| 帧数 / 每帧点数 | 154 / 20 000 |
| 全局点数 / 15 cm 体素后 | 3 054 009 / 1 253 018 |
| 轨迹长度 | 69.38 m |
| ICP 中位最近邻残差 | 0.127 m |
| 融合全程 GPU 耗时 | 18.7 s |
| 渲染 | 154 帧 / 20 fps / 1280×720 / 5.5 s（0.036 s/帧）/ 19.03 MB |
| 压力测试 | 2005 帧 × 305 万点 / 25.3 s → **2.42 亿点/秒**，GPU 平均 89.6%、峰值 93% |

### 3.5 v1 最重要的一条教训

**两条轨迹必须先放进同一个坐标系再算误差。** ICP 链表达在"第 0 帧激光系"（x 轴指向当时车头），
RTK 位置在 ENU 类世界系（x 轴指向东），两者差一个固定旋转。直接相减会得到：

| 处理方式 | 纯 ICP 里程计 ATE 均值 | 末帧 |
|---|---|---|
| 不做对齐（**错**） | 31.76 m | 69.71 m |
| 用第 0 帧 INS 姿态旋转后（对） | 0.61 m | 1.20 m |

对齐用的是第 0 帧 INS 姿态，**不是**"事后拟合的最优旋转"——后者会把误差藏进对齐矩阵里。
这个坑在 v2 里变成了 `frame0` 约定写进 `navstack/kitti.py`：所有位姿都是 `T_frame0_body`，
下游不需要任何额外变换。

另一条：**步长均值可以用来快速判错**。里程计步长 0.45 m 对 RTK 步长 0.45 m，说明尺度没崩；
ATE 却很大，那就说明方向/坐标系错了，而不是配准坏了。这个对照比盯着 ATE 更快定位问题。

---

## 4. 版本二（v2）：六主题大地图导航栈

### 4.1 六个主题 ↔ 模块映射

| # | 主题 | 主要模块 | 验收脚本 | 结果文件 |
|---|---|---|---|---|
| ① | 建图实时存储 | `store.py`、`schema.py`、`tiles.py` | `run_stage2_map.py` | `stage2_map.json` |
| ② | 地图按需加载 | `tiles.py`、`store.py` | `run_stage2_map.py` | `stage2_map.json` |
| ③ | 视觉层 | `vision.py` | `run_stage3_vision.py` | `stage3_vision.json` |
| ④ | 多楼层 | `levels.py` | `run_stage4_levels.py` | `stage4_levels.json` |
| ⑤ | 自动重定位 | `relocal.py` | `run_stage5_relocal.py` | `stage5_relocal.json` |
| ⑥ | 大地图全局导航 | `nav.py` | `run_stage6_nav.py` | `stage6_nav.json` |
| – | 前端与因子图（地基） | `lio.py`、`imu.py`、`graph.py` | `run_stage1.py` | `stage1_ablation.json` |

### 4.2 `v2/navstack/` —— 库代码，13 个文件

| 文件 | 行数 | 职责 | 关键知识点 |
|---|---|---|---|
| `__init__.py` | 22 | 包导出与主题映射 | – |
| `kitti.py` | 243 | KITTI raw 读取器：标定链、点云、相机、OXTS、IMU | `Tr_velo→cam` 链式变换、Mercator→局部平面、**`frame0` 约定（IMU body，x 沿车头、z 沿实测重力）** |
| `bundle.py` | 76 | 加载 numpy 包、把位姿序列转成关键帧对象的公共管道 | 数据契约；避免每个 stage 各写一份加载逻辑 |
| `lio.py` | 233 | 前端：GPU 上的帧间 **point-to-plane ICP** | 法向估计、点到面残差、近邻搜索、`build_reg_cloud` 的配准点云构造 |
| `imu.py` | 209 | IMU 预积分封装 | `PreintegratedImuMeasurements`、bias 随预积分线性传播、重力对齐 |
| `graph.py` | 338 | GTSAM 因子图：LIO 相对因子 + RTK 平移先验 + IMU 因子 + 回环 | `PoseTranslationPrior3D`、`ImuFactor`（单共享 bias key）、`BetweenFactorConstantBias`、**Huber 核**、Levenberg–Marquardt |
| `schema.py` | 181 | 地图磁盘格式的"契约"：map 目录里每个文件叫什么、什么类型、什么版本 | schema 版本化；生产者与消费者解耦 |
| `store.py` | 462 | 增量、崩溃安全、可按区域载入的地图存储 | **原子写**（临时文件 + rename）、**journal + 游标**、flush 批处理、按 chunk 载入、LRU 缓存 |
| `tiles.py` | 135 | 空间分块与 LOD | 网格切块、三级 LOD 叶子尺寸（1.0 / 0.4 / 0.2 m）、"大地图只有能被按需取出才打得开" |
| `levels.py` | 189 | 多楼层：按地面高度分层、按层烘焙、按层加载 | 楼层间隙判定、每层独立烘焙、全局小地图、**"硬分是错的"这份自觉**（真实数据只有 1 层就报 1 层） |
| `vision.py` | 366 | 视觉层：动态实例掩膜、语义代价层、立体深度、视觉描述子 | 实例分割（Mask R-CNN）→ LiDAR 点剔除、HSV/Tiny 描述子召回、语义代价只作用在 free 空间 |
| `relocal.py` | 319 | 重定位：先便宜的召回，后昂贵的确定 | **Scan Context**（环 × 扇区，逐扇区余弦）、地面相对高度裁剪、L1 top-k → L2 多假设 ICP 验证、视觉投票融合 |
| `nav.py` | 689 | 分层导航：全局轻量层 + 滚动窗口细层 + 分层 A* | **射线投射建占位栅格**、`GroundField`、**自车足迹清除**、占位距离门限、膨胀的**分辨率地板**、分层 A*、碰撞约束平滑、`snap_to_free`（EDT） |

### 4.3 `v2/scripts/` —— 可执行入口，18 个文件

**主流程（一阶段一个）**

| 文件 | 行数 | 做什么 | 产出 |
|---|---|---|---|
| `prep.py` | 170 | KITTI raw → 紧凑 numpy 包 | `data/*.npy` |
| `run_stage1.py` | 359 | 配准 + 因子图，含 **9 组消融**（RTK / IMU / 退化注入） | `stage1_ablation.json` |
| `run_stage2_map.py` | 234 | 逐帧写地图，再按区域读回：延迟、LOD、缓存、崩溃恢复 | `stage2_map.json` |
| `run_stage3_vision.py` | 287 | 视觉层，三个角度量：掩膜、配准提升、描述子召回 | `stage3_vision.json` |
| `run_stage4_levels.py` | 163 | 楼层分割与按层加载（真实 1 层 + 合成双层） | `stage4_levels.json` |
| `run_stage5_relocal.py` | 316 | L1 检索召回、L2 验证精度、旋转不变性、留出区误接受 | `stage5_relocal.json` |
| `run_stage6_nav.py` | 456 | 粗层 + 细层 + 分层 A*，对比单层 0.25 m 基线，含规模缩放 | `stage6_nav.json` |
| `run_all.py` | 116 | 按顺序跑完六段 + 两段可视化，汇总成 `summary.json` | `out/summary.json` |

**可视化**

| 文件 | 行数 | 做什么 |
|---|---|---|
| `export_web.py` | 516 | 把点云、占位栅格、三条规划路径与六段指标**内嵌**进单个 HTML（WebGL2） |
| `render_video.py` | 262 | 鸟瞰地图生长 + 前视相机双画面，HUD 叠加逐帧误差 |

**诊断脚本（都是"先用真实命令查证"的产物，不是装饰）**

| 文件 | 行数 | 回答什么问题 |
|---|---|---|
| `diag_imu_agreement.py` | 158 | LIO 增量与 IMU 预积分增量是否自洽？ |
| `diag_imu_bias.py` | 163 | OXTS 加速度能不能用？bias 多大？（**结论作废**，留档记录失败） |
| `tune_bias.py` | 132 | bias 先验 × frame0 姿态 12 格扫描，证明 frame0 倾角不该从加速度计估 |
| `tune_imu.py` | 170 | 预积分本身错了，还是只是权重错了？ |
| `diag_nav_layer.py` | 173 | 为什么 0.25 m 单层基线拒绝规划？（ego 格取值、隔离率、A* 成败） |
| `diag_nav_ground.py` | 188 | 是不是**地面**噪声进了带？（→ 证伪） |
| `diag_nav_points.py` | 127 | 车开过的格子里到底是哪些点、来自哪一帧、离车多远？ |
| `diag_nav_sweep.py` | 122 | 占位距离门限 × 自车足迹 × 膨胀 的消融，用测量选参数 |

### 4.4 `v2/results/` —— 14 份结果

| 文件 | 内容 | 文中对应 |
|---|---|---|
| `stage1_ablation.json` | 9 组消融的 ATE / RPE / 段内 ATE | 前端与因子图 |
| `tune_bias.json` | bias 先验与 frame0 姿态的 12 格扫描 + 诊断 | frame0 约定 |
| `imu_sweep.json` | IMU 噪声参数扫描 | 因子权重 |
| `imu_lio_agreement.json` | LIO 与 IMU 增量自洽性 | 前端可信度 |
| `imu_bias_fit.json` | bias 反解（**结论作废**） | 失败留档 |
| `stage2_map.json` | 写入延迟、flush、LOD 载入、缓存、崩溃恢复 | 主题 ①② |
| `stage3_vision.json` | 动态掩膜、配准提升、描述子、立体深度、语义代价 | 主题 ③ |
| `stage4_levels.json` | 真实/合成双层、按层加载节省、全局小地图 | 主题 ④ |
| `stage5_relocal.json` | L1/L2 指标、旋转不变性、视觉投票、先验精化、留出区 | 主题 ⑤ |
| `stage6_nav.json` | 分层 vs 单层、规模缩放、代表性路径 | 主题 ⑥ |
| `stage6_nav_rebuild.json` | 消融：窗口层"载入"vs"重建" | 主题 ② 与 ⑥ 的耦合 |
| `diag_nav_layer.json` | 各分辨率下 ego 格与邻格取值 | 排障 |
| `diag_nav_ground.json` | 相对局部地面的高度分布（证伪地面假设） | 排障 |
| `diag_nav_sweep.json` | 距离门限 × 足迹 × 膨胀 消融 | 排障 |

### 4.5 关键结果

**① 因子图与 IMU 的价值（`stage1_ablation.json`）**

| 场景 | 无 IMU | 有 IMU |
|---|---|---|
| RTK 正常 | 0.076 m | 0.086 m |
| RTK 失锁 40–90 帧 | **0.192 m** | 0.195 m |
| RTK 长失锁 30–120 帧 | **0.197 m** | 0.297 m |
| **LiDAR + RTK 同时退化 60–90 帧** | 0.952 m | **0.069 m（13.8×）** |

> 结论：保留 IMU 因子，但不要指望它在 GPS 短暂丢星时改善精度。它买的是**退化场景的可用性**，
> 不是常态精度。

**② 建图存储与按需加载（`stage2_map.json`）**

| 指标 | 数值 |
|---|---|
| 逐帧写入 | 均值 12.61 ms / p95 14.9 ms / 峰值 30.71 ms（帧周期 103.3 ms） |
| flush | 4 次合计 1423.1 ms / 138 个文件 |
| 磁盘 | 38.13 MB / 154 关键帧 |
| 热载入（区域 vs 全图） | coarse 0.09 ms、mid 0.34 ms、**fine 0.87 ms** 对全图 704–742 ms（点占比 7.51%） |
| 冷载入 | 0.86 / 2.30 / **6.07 ms** |
| 缓存命中率 | 0.593 |
| 崩溃恢复 | 62/62 |

**③ 视觉层（`stage3_vision.json`）**

| 指标 | 数值 |
|---|---|
| 动态实例检出 | **1.000**（154/154 帧，2239 个实例，掩膜像素占 10.0%） |
| 配准残差 | 0.0840 → **0.0824 m（改善 1.8%）** |
| 立体深度中位 | 0.14–0.55 m（76–86% 在 1 m 内） |
| 描述子 R@1 | HSV **0.9481** / Tiny 0.7468 |
| 语义代价 | free 空间 4.2–7.5% |

**④ 多楼层（`stage4_levels.json`）**

| 指标 | 数值 |
|---|---|
| 真实数据 | **1 层**（正确拒绝硬分，z_ref −1.715 m） |
| 合成双层 | 2 层 / 分配准确率 **1.000** / 省 **49.5%** 点 |
| 全局小地图 | [205, 174] @1.0 m，观测 9098 m²，占据 7163 格 |

**⑤ 重定位（`stage5_relocal.json`）**

| 指标 | 数值 |
|---|---|
| 44 查询 L1 | **R@1 0.7955 / R@5 0.8864**，115.8 ms |
| 旋转 ±90°/180° | 全 **1.000** |
| 留出区误接受 | **0%** |
| 视觉投票 | 0.7955 → **0.864** |
| L2 ICP 残差 | 0.1252 m |
| 带里程计先验的精化 | 中位 **0.017 m** / p90 0.035 m（无先验时 0/44） |

**⑥ 分层导航（`stage6_nav.json`）**

| 指标 | 分层 | 单层 0.25 m 基线 |
|---|---|---|
| 成功率（10 对） | **1.000** | 1.000 |
| 冷启动 | **1357.6 ms** | 2295.7 ms |
| 热启动 | **83.4 ms** | 2295.7 ms |
| A* 展开格数 | **52** | **1203（×22.4）** |
| 路径长度 | 43.09 m | 43.14 m（**−0.36%**） |
| 最小净空 | 0.25 m | 0.25 m |
| 落障碍格 | **0.00%** | 0.00% |

**加速比：冷启动 ×1.69，热启动 ×27.6。** 计时指标有约 ±5% 运行抖动。

**消融（`stage6_nav_rebuild.json`）**：把窗口层从"载入 chunk"换成"重新射线投射"后，热启动变成
4219.7 ms（×0.53，**比单层还慢约 2 倍**）。这说明主题 ② 与 ⑥ 的耦合点是**"载入"而不是"重建"**
——分层导航只有在细层可被载入时才成立。

### 4.6 第六主题踩过的三个真 bug（都已修，都有数字）

1. **0.25 m 细层把"车开过的路面"判成障碍，单层基线 10/10 全败（`expanded=1`）。**
   点级探针发现 ego 所在 0.25 m 格内有 **44 个带内回波**，dz +0.20~+1.72 m、来自约 16 帧、
   距车 0.1–0.5 m——是**自车本体回波**，任何距离门限都删不掉。
   修复：`clear_footprint()` 按半车宽 0.80 m 圆盘清除并置 FREE，**且膨胀后必须再次清除**。
2. **被证伪的假设**：「地面噪声进带」。实测旧带只有 **1.1%** 落在新地面以上 0.2 m 以下；
   换局部地面场后带内点数反而 38.8% → 66.9%。这条假设不成立，记录在案但未采用。
3. **跨分辨率拼接顶点落在障碍格里（8.26%）。** 给窗口层加 0.25 / 0.50 / 0.75 m 膨胀余量
   **三次都是 8.26%**（无效），证明问题在**粗层格心航点**而非窗口层。正解是 `snap_to_free()`
   用 EDT 找最近自由格（平均修 7.1 个顶点），修后 0.00%。

---

## 5. 知识点索引

按"概念 → 一句话 → 在哪实现/验证"排列，方便按需跳读。

### 5.1 状态估计

| 概念 | 一句话 | 位置 |
|---|---|---|
| **point-to-point ICP** | 最小化对应点对的欧氏距离平方；实现简单，但在平滑曲面上收敛慢、残差偏大 | v1 `fuse/fuse.py` |
| **point-to-plane ICP** | 最小化点到目标点**切平面**的距离，沿法向收敛，曲面上一阶即准；残差 0.127 → 0.084 m | v2 `navstack/lio.py` |
| **`frame0` 约定** | 所有位姿 = `T_frame0_body`，frame0 为 IMU body 系，x 沿行驶方向、z 沿实测重力；旋转保持单位阵 | v2 `navstack/kitti.py` |
| **加速度计估重力是陷阱** | 车全程未停（最低 2.54 m/s、最小 \|gyro\| 0.40 °/s），没有静止段；三种估法倾角互差 6.3° | v2 `tune_bias.json` |
| **IMU 预积分** | 把两帧之间的 IMU 采样压成一个相对运动约束，避免在优化里重复积分；bias 变化用一阶雅可比修正 | v2 `navstack/imu.py` |
| **`ImuFactor` 单共享 bias key** | 相邻两帧共用一个 bias 状态，避免每帧一个 bias 导致不可观 | v2 `navstack/graph.py` |
| **Huber 鲁棒核** | 残差小于 k 时二次、大于 k 时转线性，抑制 RTK 跳点；`rtk_huber_k=1.5` | v2 `navstack/graph.py` |
| **ATE / RPE / 段内 ATE** | 绝对轨迹误差 / 相对位姿误差 / **只在失锁窗口内**统计的 ATE。分窗 ATE 才是揭示 IMU 价值的指标 | v2 `run_stage1.py` |
| **常速度初值** | 用上一步速度外推当前位姿作为 ICP 初值，减少迭代并避免局部极小 | v1 `fuse/fuse.py` |

### 5.2 建图与存储

| 概念 | 一句话 | 位置 |
|---|---|---|
| **原子写** | 先写临时文件再 `rename`，避免写到一半断电留下半截文件 | v2 `navstack/store.py` |
| **journal + 游标** | 每次写入追加一条日志并推进游标；崩溃后重放日志补齐缺失帧（实测恢复 62/62） | v2 `navstack/store.py` |
| **分块（chunk）** | 按 30 m 网格切块，载入时只取附近 chunk | v2 `navstack/tiles.py` |
| **LOD** | 三级叶子尺寸 1.0 / 0.4 / 0.2 m；粗层 12 k 点、细层 465 k 点，同一份数据多种精度 | v2 `navstack/tiles.py` |
| **chunk 是派生物** | 由关键帧可重新生成，损坏可重建而不必重跑建图 | v2 `store.py`、`stage2_map.json` |
| **schema 契约** | 磁盘格式显式版本化，生产者与消费者解耦 | v2 `navstack/schema.py` |

### 5.3 视觉层

| 概念 | 一句话 | 位置 |
|---|---|---|
| **动态实例掩膜** | 用实例分割找出可动类别（车/人/骑行者），把它们对应的 LiDAR 点从建图里剔除 | v2 `navstack/vision.py` |
| **语义代价层** | 只对 **free** 空间加代价（草地/人行道贵一点），不改变占据格 | v2 `vision.py` |
| **立体深度** | 双目基线 0.5327 m 反投影，验证掩膜区域的空间一致性 | v2 `vision.py` |
| **视觉描述子召回** | HSV 直方图（144 维）R@1 0.9481 vs Tiny 特征 0.7468；这段同光照数据上简单描述子更好 | v2 `vision.py` |
| **视觉投票** | 把描述子相似度融进 L1 候选打分，R@1 0.7955 → 0.864 | v2 `relocal.py` |

### 5.4 重定位

| 概念 | 一句话 | 位置 |
|---|---|---|
| **Scan Context** | 把一圈点云按"环 × 扇区"编码成二维描述子，对旋转用**逐扇区余弦**取最优列移 | v2 `navstack/relocal.py` |
| **地面相对高度裁剪** | 只保留地面以上一定高度的点再做描述子，避免地面主导（裁剪高度取错会把地面全裁掉，需按数据标定） | v2 `relocal.py` |
| **L1 / L2 分离** | L1 用便宜描述子取 top-k（115.8 ms），L2 用 ICP 多假设验证（4227 ms）。便宜的先做 | v2 `relocal.py` |
| **留出区检验** | 把 60–95 帧从数据库删掉再查询，看会不会"硬认"出错误位置。实测误接受 0% | v2 `run_stage5_relocal.py` |
| **先验精化** | 有里程计先验时只做局部精化，中位 0.017 m；无先验时全局搜索 0/44 成功 | v2 `relocal.py` |

### 5.5 占位栅格与导航

| 概念 | 一句话 | 位置 |
|---|---|---|
| **射线投射建图** | 从扫描原点向每个回波投射射线，射线穿过的格标 free、端点格标 occupied | v2 `navstack/nav.py` |
| **`GroundField`** | 2 m 栅格、每格取 z 的低分位（p2）、空洞最近邻填充、中值滤波；用来算"相对局部地面的高度" | v2 `nav.py` |
| **z 带选择** | 只把地面以上 0.20–2.20 m 的回波当障碍候选 | v2 `nav.py` |
| **自车足迹清除** | 车体占过的格子按定义可通行；按半车宽 0.80 m 圆盘清成 FREE，**膨胀后必须再清一次** | v2 `nav.py` |
| **占位距离门限 `max_occ_m`** | 回波离自身扫描原点超过 30 m 不许制造障碍（自由空间仍投射到 35 m）。单独用无效，与足迹清除合用才生效 | v2 `nav.py` |
| **膨胀的分辨率地板** | 膨胀按格数算：0.60 m 在 1.0 m 栅格上等于 **0 格**（实测膨胀增加 0.0 pp），在 0.25 m 上是 3 格（+7.7 pp） | v2 `nav.py` |
| **分层 A*** | 先在 1.0 m 粗层求通路，再在窗口内 0.25 m 细层精化。展开格数 52 vs 1203 | v2 `nav.py` |
| **滚动窗口** | 沿粗路径取 30 m 窗口（半径 18 m + 余量），逐窗细化 | v2 `nav.py` |
| **碰撞约束平滑** | 平滑时惩罚落入障碍格的顶点；无约束 Laplacian 会削角，最小净空恒为 0.00 m | v2 `nav.py` |
| **`snap_to_free`（EDT）** | 跨分辨率拼接的顶点落在障碍格时，用欧氏距离变换找回最近自由格（平均修 7.1 个顶点） | v2 `nav.py` |
| **窗口层两种来源** | `store`（载入 chunk，毫秒级）vs `rebuild`（重跑射线投射，秒级）。这是主题 ② 与 ⑥ 的耦合点 | v2 `run_stage6_nav.py` |

### 5.6 可视化与工程

| 概念 | 一句话 | 位置 |
|---|---|---|
| **z-buffer 遮挡** | 每像素取最近深度，遮挡是**解出来的**而不是排序近似的 | v1 `render/render_video.py` |
| **`scatter_reduce_(amin)`** | 按像素索引做"取最小深度"归约，一步完成 z-buffer | v1 `render/render_video.py` |
| **超采样 + splat** | 2× 渲染后按 2×2 splat 再 `INTER_AREA` 降采样；1 像素点在高密度下就是噪点 | v1 `render/render_video.py` |
| **自包含查看器** | 点云/栅格/路径/指标全部内嵌进单个 HTML，WebGL2 直接画，无需服务器 | v2 `web/index.html` |
| **注记层要关深度测试** | 13 万点云会把路径线埋掉，路径必须作为 `DEPTH_TEST` 关闭的注记层绘制 | v2 `export_web.py` |
| **`np.tolist()` 的嵌套对** | 给的是 `[[x,y],...]`，JS 按扁平数组读会得到 NaN——draw 成功但不产生片元 | v2 `export_web.py` |
| **ffmpeg 的 SIGTTIN** | 在 pty 下 ffmpeg 读 stdin 取交互键会被挂起（`State: T`），必须 `-nostdin` + `stdin=DEVNULL` | v2 `render_video.py` |

---

## 6. 复现

### 6.1 v1

```bash
# 0. 依赖：python3 + numpy / torch(cuda) / opencv-python / pillow，以及 ffmpeg
# 1. 取数据（约 646 MB，脚本带断点续传与多连接加速）
bash fetch/fetch_kitti.sh /path/to/raw
# 2. 预处理
python prep/prep_kitti.py --raw /path/to/raw --out data
# 3. 融合
python fuse/fuse.py --data data --out out
# 4. 渲染
python render/render_video.py --data data --out out --media media
```

ICP 前端选型对照：`python fuse/icp_tune.py --data data`。

### 6.2 v2

```bash
pip install -r v2/requirements.txt

# 1) 从 KITTI 原始数据生成中间张量
python v2/scripts/prep.py --raw /path/to/kitti_raw --out v2/data

# 2) 全链（六阶段 + 两段可视化）
python v2/scripts/run_all.py --data v2/data --raw /path/to/kitti_raw --out v2/out

# 或只跑某几段
python v2/scripts/run_all.py --only stage6_nav
python v2/scripts/run_all.py --skip stage3_vision,stage5_relocal
```

> `run_all.py` 按阶段声明转发参数（例如只有 stage3 需要 `--raw`）。失败阶段会写进
> `summary.json` 的 `failed` 字段并以非零码退出——一个"静默跳过失败阶段"的调度脚本
> 比没有调度脚本更危险。

诊断脚本清单见 4.3 的第三张表。

---

## 7. 环境与实测状态

| 项 | 值 |
|---|---|
| GPU | NVIDIA GB10（DGX Spark） |
| Python | 3.12.3 |
| torch | 2.14.0+cu130（CUDA True） |
| gtsam | ROS Jazzy 绑定 |
| numpy / OpenCV | 2.5.3 / 5.0.0 |

**v2 全链实测（2026-11-15）**：`run_all.py` 六个阶段与两段可视化**全部通过**，退出码 0。

| 阶段 | 耗时 |
|---|---|
| stage1 配准 + 因子图消融 | 11.0 s |
| stage2 地图存储 | 17.1 s |
| stage3 视觉层 | 69.2 s |
| stage4 多楼层 | 18.2 s |
| stage5 重定位 | 392.9 s |
| stage6 分层导航 | 13.5 s |
| viz_web / viz_video | 4.9 s / 20.0 s |

> 首次整链运行时 stage3 与 stage5 因**同机其它负载占满内存**（116/121 GB，swap 满）连 CUDA
> 上下文都无法创建（`cuDevicePrimaryCtxRetain` OOM）而失败，与本研究代码无关；内存缓解后
> 重跑同一命令即通过。如实记录在这里，因为"能跑通"和"在什么条件下跑通"是两件事。

---

## 8. 已知局限

**v1**

1. 没有 IMU 预积分、没有回环，位姿图是 3 自由度（平移）稀疏最小二乘，不是 gtsam/ceres。
2. 旋转不做图优化：重建地图时的朝向直接取 RTK/INS 的相对姿态。
3. 权重手调（8:1），没按噪声协方差加权，也没有鲁棒核。
4. ICP 用体素/随机降采样到几千点，精度够做演示，不够做评价。
5. "RTK 中断 40–90 帧"是**人为模拟**的，不是真实失锁。

**v2**

1. 地图范围只有 61.8 × 17.1 m，**这不是"大地图"**。分层导航的收益在这个尺度上被低估：
   全图载入只要 0.7 s，分层的冷启动优势（×1.69）远不如热启动（×27.6）明显。
2. 多楼层只有 1 层真实数据，双层结论来自**合成**（把同一段点云按 3.6 m 抬高）。
3. 视觉层用通用实例分割权重，没有在本数据上微调；立体深度只用于验证，没有进因子图。
4. Scan Context 的 L2 在**没有**里程计先验时 0/44 成功——它不是能独立工作的全局重定位，
   而是"在有先验的前提下把先验修准"。
5. `frame0` 的倾角不可观：车全程未停，加速度计给不出可信重力方向，最终旋转取单位阵。

---

## 9. 引用与许可

- **代码**：MIT（见 `LICENSE`）
- **数据**：KITTI，**CC BY-NC-SA 3.0**，仅限非商业用途。原始数据不随本仓库分发。使用请引用：

```
@inproceedings{Geiger2012CVPR,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title  = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  booktitle = {CVPR}, year = {2012}
}
```

引用本数据存储库：

```bibtex
@misc{lizipeng2026navstackv2,
  author       = {Zipeng Li},
  title        = {navstack v2: LiDAR + RTK + IMU mapping and large-map navigation stack
                  --- code and experimental data},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/lemonbaby2/rtk-lidar-fusion-demo}},
  note         = {Experiment code in \texttt{v2/}; all quantitative results in
                  \texttt{v2/results/}}
}
```

若期刊要求 DOI 而非仓库链接，可在 Zenodo 对本仓库打一个 release 以获得 DOI。

## 相关

- 论文与配套代码：`SourceAware-Dynamic-Gaussian-Mapping`（动态环境下视觉语言模型辅助三维高斯语义建图与机器人目标导航）
- Gitee 镜像：<https://gitee.com/li-zipeng123/rtk-navstack-v2>
