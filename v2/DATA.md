# 数据与代码存储库说明 / Data & Code Repository

本文件说明本仓库中保存了哪些研究数据、每个文件对应论文中的哪张表哪句话，
以及如何从原始公开数据集重新生成它们。

**仓库地址**：<https://github.com/lemonbaby2/rtk-lidar-fusion-demo>
（实验本体在 `v2/` 目录；镜像：<https://gitee.com/li-zipeng123/rtk-navstack-v2>）

---

## 1. 一句话概括

仓库保存的是**本研究自己产生的全部数据**：六个主题的实验结果（`results/*.json`）、
可复查的派生可视化（自包含 WebGL 查看器与视频）、以及生成它们的全部代码。
**原始 KITTI 传感器数据不随仓库分发**（见第 5 节的许可说明），可用仓库内的脚本从
KITTI 官方站点重新下载并预处理。

---

## 2. 结果数据清单（`results/`，数字的唯一来源）

论文中出现的每一个数字都能在这些 JSON 里按字段查到。均为脚本直接写出的原始结果，
没有手工填写或后期修饰。计时类指标是同一台机器上的实测中位数，多次运行间有约 ±5% 抖动。

### 2.1 前端与因子图（论文第 3 节）

| 文件 | 内容 | 关键字段 |
|---|---|---|
| `stage1_ablation.json` | 9 组消融（RTK/IMU/退化注入）的 ATE、RPE、段内 ATE | `ablation[].name` / `.ate_mean_m` / `.ate_rmse_m` / `.rpe1_m` / `.in_gap_ate_m`；`best` |
| `tune_bias.json` | IMU bias 先验 × frame0 姿态 12 格扫描与诊断 | `reference`、`sweep[]`、`diagnosis.tilt_estimates_deg`、`diagnosis.vehicle_never_stops` |
| `imu_sweep.json` | IMU 噪声参数扫描 | `rows[]`、`best` |
| `imu_lio_agreement.json` | LiDAR 增量与 IMU 预积分增量的自洽性检验 | `rotation_diff_deg`、`translation_diff_m`、`verdict` |
| `imu_bias_fit.json` | bias 反解尝试（**结论作废**，留档以记录失败） | `zero_bias`、`after_bias_linear`、`implied_tilt_deg` |

> `imu_bias_fit.json` 的方法有已知缺陷（灵敏度矩阵漏掉速度项二阶项，反解出 439 m/s² 的
> 荒谬 bias），**不作为结论使用**，保留它是为了让失败的尝试也可复查。

### 2.2 建图与存储（论文第 4 节）

| 文件 | 内容 | 关键字段 |
|---|---|---|
| `stage2_map.json` | 逐帧写入延迟、flush、分块 LOD 载入、缓存、崩溃恢复 | `write.keyframe_ms_*`、`read.{coarse,mid,fine}.region_ms_median`、`cache_warm.hit_rate`、`recovery`、`chunks_are_derived` |

### 2.3 视觉层 / 多楼层 / 重定位 / 导航（论文第 5–8 节）

| 文件 | 内容 | 关键字段 |
|---|---|---|
| `stage3_vision.json` | 动态实例掩膜、配准残差改善、视觉描述子召回、立体深度 | `dynamic_mask.*`、`descriptor.{hsv,tiny}.*`、`stereo.*`、`semantic_cost.*` |
| `stage4_levels.json` | 楼层分割（真实/合成双层）、按层加载节省、全局小地图 | `real.*`、`synthetic_duplex.*`、`minimap_resolutions.*` |
| `stage5_relocal.json` | Scan Context L1 检索、L2 多假设验证、视觉投票、先验精化、留出区误接受 | `metrics.*`、`vision_vote.*`、`refinement_with_prior.*`、`absence_test.*` |
| `stage6_nav.json` | 分层导航 vs 单层基线、规模缩放、代表性路径 | `aggregate.*`、`scaling[]`、`overlay`（查看器用的路径） |
| `stage6_nav_rebuild.json` | 消融：窗口层「载入 chunk」vs「重建」 | `aggregate.hier_warm_ms_median`、`speedup_warm_median` |

### 2.4 诊断记录（失败与证伪的原始数据）

| 文件 | 用途 |
|---|---|
| `diag_nav_layer.json` | 各分辨率下自车所在格的栅格取值、隔离率、A* 成败 |
| `diag_nav_ground.json` | 回波相对局部地面的高度分布（用于**证伪**「地面噪声进带」假设） |
| `diag_nav_sweep.json` | 占位距离门限 × 自车足迹 × 膨胀 的消融 |

这三份是排障过程的原始观测，论文第 8 节与 `docs/RESULTS.md` 的失败表引用它们。

---

## 3. 派生可视化数据

| 路径 | 内容 |
|---|---|
| `web/index.html` | 自包含 WebGL 查看器（2.7 MB，数据全部内嵌，双击即开，无需服务器）：点云、占位层栅格、三条规划路径、六段指标面板 |
| `web/preview.png` | 查看器截图 |
| `media/navstack_v2.mp4` | 鸟瞰 BEV + 前视相机双画面视频（1760×738 / 15 fps / 154 帧 / 10.3 s），绿=RTK 真值、蓝=融合估计、红=被视觉层剔除的动态回波 |
| `media/frames/f60.png`、`f140.png` | 视频抽帧样张 |

这些不是"另画一张图"，而是由 `scripts/export_web.py` 与 `scripts/render_video.py`
直接读 `results/` 与 `out/` 生成，与表格数字同源。

---

## 4. 代码与复现

```
navstack/     库代码，每个主题一个模块（lio.py / graph.py / imu.py / store.py /
              levels.py / vision.py / relocal.py / nav.py）
scripts/      可执行入口，每阶段一个 + 六个诊断脚本
configs/      参数
tests/        单元测试
docs/RESULTS.md   完整方法与结果说明（含 13 条失败与教训）
```

一键复现：

```bash
pip install -r requirements.txt

# 1) 从 KITTI 原始数据生成中间张量（需要自行下载原始数据，见第 5 节）
python scripts/prep.py --raw /path/to/kitti_raw --out data

# 2) 全链（六个阶段 + 两段可视化），结果写入 out/
python scripts/run_all.py --data data --raw /path/to/kitti_raw --out out
```

实测环境：Python 3.12.3、torch 2.14.0+cu130（CUDA True）、gtsam（ROS Jazzy 绑定）、
numpy 2.5.3、OpenCV 5.0.0、scipy；GPU 为 NVIDIA GB10（DGX Spark）。

### 复现状态（如实记录，2026-11-15）

`run_all.py` 全链在一次完整运行中 **stage1 / stage2 / stage4 / stage6 与两段可视化全部通过**；
**stage3 与 stage5 当次失败**，原因是同机其它负载占满内存（116/121 GB，swap 已满），
连 CUDA 上下文都无法创建（`cuDevicePrimaryCtxRetain` OOM），与本研究代码无关。
这两个阶段的 `results/*.json` 来自同机内存空闲时的成功运行，代码此后未修改。
当内存压力解除后重跑同一命令即可复现。

---

## 5. 原始数据来源与许可

- **数据集**：KITTI raw `2011_09_26_drive_0005`（154 帧 / 15.8 s / 69.4 m）
- **传感器**：Velodyne HDL-64E、彩色相机 02+03（基线 0.5327 m）、OXTS RT3003
  （RTK 级 GNSS/INS，100 Hz）
- **获取方式**：<https://www.cvlibs.net/datasets/kitti/raw_data.php>
- **许可**：**CC BY-NC-SA 3.0**，仅限非商业用途。引用：

```
@inproceedings{Geiger2012CVPR,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title  = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  booktitle = {CVPR}, year = {2012}
}
```

**为什么仓库里没有原始点云/图像**：KITTI 的许可不允许我们重新分发原始或近原始的
传感器数据。仓库提供的是**本研究产生的派生结果**（`results/*.json`、可视化产物）与
**生成脚本**；下载 `scripts/prep.py` 所需的原始文件后，整条链路可在数分钟内重建。

本研究代码采用 MIT 许可（见 `LICENSE`）。

---

## 6. 如何引用本数据存储库

> 李梓鹏. *navstack v2: LiDAR + RTK + IMU 建图与大地图导航栈——代码与实验数据*.
> GitHub, 2026. <https://github.com/lemonbaby2/rtk-lidar-fusion-demo>

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

若期刊要求 DOI 而非仓库链接，可在 Zenodo 对本仓库打一个 release 标签以获取 DOI，
届时应把 DOI 一并写进论文的数据可用性声明。
