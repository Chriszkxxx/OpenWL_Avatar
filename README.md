# OpenWL-Avatar

## 目录结构

```
codes/
├── assets/              # 示例图片和 data.jsonl
├── models/              # 模型 wrapper（每个模型一个文件）
│   ├── gen_image/       # 图像生成（Qwen-Image-Edit 等）
│   ├── gen_3d/          # 3D 生成（Trellis 等）
│   ├── gen_video/       # 视频生成（HunyuanVideo 等）
│   └── reasoning/       # 视觉语言模型（Qwen-VL 等）
├── operators/           # 流水线 operator（加载 models，调用 funcs）
│   ├── gen_ue_avatar/   # UE avatar 生成流水线（纯 AI 推理）
│   │   ├── operator.py  # UEAvatarOperator
│   │   └── funcs/       # 解耦功能函数
│   ├── gen_game_cg/     # 游戏 CG 生成流水线
│   │   ├── operator.py  # GameCGOperator
│   │   └── funcs/
│   └── process_input/   # 输入预处理流水线
│       ├── operator.py  # InputProcessor
│       └── funcs/
├── serving/             # HTTP serving + UE5 RPC 交付层
│   ├── server.py        # HTTP API 入口（接收前端请求）
│   └── ue_client.py     # UE5 RPC 调用（推送结果到 UE5）
└── test/                # 测试入口（调用尽可能简单）
```

## 调用逻辑

```
输入请求 (image + text)
     ↓
serving/server.py        ← 接收前端请求
     ↓
InputProcessor           ← operators/process_input/operator.py
     ↓
UEAvatarOperator         ← operators/gen_ue_avatar/operator.py（纯 AI 生成）
  ├── gen_tpose()        ← funcs/gen_tpose.py        (GenImageModel)
  ├── gen_3d_avatar()    ← funcs/gen_3d_avatar.py    (Gen3DModel / Trellis)
  ├── rig_avatar()       ← funcs/rig_avatar.py       (PuppeteerModel: 骨骼+蒙皮)
  ├── retarget_motion()  ← funcs/retarget_motion.py  (world-delta 动作重定向)
  └── gen_motion()       ← funcs/gen_motion.py       (rig + 可选 retarget)
     ↓
serving/ue_client.py     ← Python 调 UE5 RPC，推送结果到前端展示

GameCGOperator           ← operators/gen_game_cg/operator.py
  ├── gen_storyboard()   ← funcs/gen_storyboard.py (ReasoningModel)
  ├── gen_cg_video()     ← funcs/gen_cg_video.py   (GenVideoModel)
  └── compose_cg()       ← funcs/compose_cg.py
```

## 快速开始

```python
# 1. 加载 operator（内部自动加载所需模型）
from operators.gen_ue_avatar.operator import UEAvatarOperator
from serving.ue_client import import_avatar_to_ue, import_motion_to_ue

op = UEAvatarOperator({
    "gen_image_model": "/path/to/Qwen-Image-Edit",
    "gen_3d_model":    "/path/to/TRELLIS",
    "device":          "cuda",
})

# 2. 运行 AI 生成流水线
from PIL import Image
ref = Image.open("assets/luffy.jpg")

result = op.run(ref, description="straw hat pirate", motion_desc="walk forward")

# 3. 推送到 UE5（由 serving 层负责）
import_avatar_to_ue(result["mesh_path"])
import_motion_to_ue(result["motion_path"], avatar_name="Avatar")
```

## 骨骼绑定 (rigging) 与动作重定向 (retarget)

Puppeteer 作为 `gen_3d` 下的「监测骨骼 + 蒙皮」模型接入，并附带一个纯 bpy 的动作
重定向引擎：

```
models/gen_3d/
├── puppeteer.py                 # PuppeteerModel：rig() 骨骼+蒙皮，retarget() 动作重定向
├── puppeteer_retarget/          # 已提交的 bpy 重定向引擎（world-conjugation-delta）
│   ├── rig_io.py                # 解析 rig.txt、建骨架、蒙皮权重、导入 GLB
│   ├── world_delta.py           # 源动作 (FBX/BVH) -> Puppeteer 世界增量重定向（推荐）
│   ├── bvh_to_mixamo.py         # BVH -> Mixamo FBX（旧两步工作流，保留作 fallback）
│   └── mappings/*.json          # 骨骼映射表（含 BVH->Puppeteer 直接映射）
└── Puppeteer_main/              # Puppeteer 源码（由 install_puppeteer.sh 克隆，不提交）
```

安装与运行：

```bash
bash scripts/installing/install_puppeteer.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"   # bpy 需要 X11/OpenGL 库

python test/test_rigging.py     # GLB -> Puppeteer rig（骨骼 + 蒙皮）+ bind-pose FBX
python test/test_retarget.py    # rig + Mixamo FBX -> 动画 FBX（mesh+anim 与 anim-only）
```

代码调用：

```python
op = UEAvatarOperator({"puppeteer_root": "models/gen_3d/Puppeteer_main", "device": "cuda"})

rig = op.rig_avatar("output/avatar.glb")                       # 1. 自动绑定骨骼
op.retarget_motion("output/avatar.glb", rig["rig_txt"],        # 2a. Mixamo FBX 动作
                   "Slow Run.fbx", source="mixamo", fps=30)
op.retarget_motion("output/avatar.glb", rig["rig_txt"],        # 2b. BVH 直接重定向
                   "motion.bvh", source="bvh", fps=20)         #     (无需 Mixamo 中转)
```

> 说明：重定向采用 world-space 旋转增量，对源/目标骨骼的局部 roll 不敏感，可避免手臂
> 漂移 / 膝盖反向等问题。正因为只传递世界旋转，BVH（如 MoMask）可**直接**重定向，无需
> 先转成 Mixamo FBX；源类型按文件后缀自动识别，根骨骼从 mapping 的 `root_bones` 读取。
> 若 BVH 单位与骨架不一致，可用 `global_scale` / `root_scale` 调整根位移。
> `anim_only` FBX 适合在 UE 中以 "Existing Skeleton" 导入动作。headless 下 bpy 在进程
> 退出时可能 segfault（无害）；只要 FBX 已写出即视为成功。旧两步法仍可用
> `source="bvh_via_mixamo"` + `mixamo_ref=<同体型 Mixamo FBX>` 触发。

## 扩展新功能

1. 在对应 `funcs/` 目录下新建 `.py` 文件，实现函数
2. 在 `operator.py` 中添加调用方法
3. 在 `models/` 对应子目录添加 model wrapper
4. 在 `test/` 添加测试文件
