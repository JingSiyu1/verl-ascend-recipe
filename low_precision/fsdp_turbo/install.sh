#!/bin/bash
# Qwen3.5-35B-A3B MXFP8 GRPO（FSDPTurbo 链路）环境安装脚本
# 用法: bash install.sh（在本机工作目录下执行，依赖仓会克隆到当前目录）
set -ex

echo "1. install vllm"
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 0fc695f
VLLM_TARGET_DEVICE=empty pip install -e . --no-deps --no-build-isolation
cd ..

echo "2. install vllm-ascend"
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout f16cfd1
pip install -r requirements.txt
export COMPILE_CUSTOM_KERNELS=1
pip install -e . --no-deps --no-build-isolation
cd ..

echo "3. install FSDPTurbo"
# 版本要求见 low_precision/fsdp_turbo/readme.md「版本依赖」节；下方 pin 为已验证基线。
git clone https://gitcode.com/Ascend/FSDPTurbo.git
cd FSDPTurbo
git checkout b1793de
pip install -e .
cd ..

echo "4. install verl"
# 版本要求见 low_precision/fsdp_turbo/readme.md「版本依赖」节。
# 过渡说明：本链路依赖的 verl 修正（vLLM 量化权重名映射
# translate_weight_names_for_vllm、FSDP 模型合并器 mesh/placement 兼容）
# 尚未合入 verl 主仓，过渡期使用 fork 分支 JingSiyu1/verl@fsdpturbo-ascend-fixes
# （含 afdead79 / 0f6df719）。主仓合入后请改回主仓 pin 并补记 commit 号。
git clone https://github.com/JingSiyu1/verl.git
cd verl
git checkout fsdpturbo-ascend-fixes
sed -i '/^triton-ascend==/ s/^/# /' requirements-npu.txt
pip install -r requirements-npu.txt
cd ..

echo "5. install triton-ascend && transformers"
pip install transformers==5.10.4
pip install triton-ascend==3.2.2 --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn

echo "6. check required features (hard fail)"
# 本链路硬性依赖以下特征，缺失时 MXFP8 链路会静默产出错误结果
# （2026-10-20 e2e v1：verl 无 translate_weight_names_for_vllm 时 2056 个
# 权重名映射失败、rollout 全乱码、step0 val 0/601），故自检失败即退出，
# 请改用 readme「版本依赖」节指定的版本：
#   verl      : translate_weight_names_for_vllm（vLLM 量化权重名映射）
#   FSDPTurbo : quant_ignored_modules（训练侧排除 linear_attn 依赖此项）
fail=0
if ! grep -rq --exclude-dir=.git "translate_weight_names_for_vllm" verl/; then
    echo "ERROR: 未在 verl 中找到 translate_weight_names_for_vllm ——"
    echo "       量化权重名将无法映射，推理侧会静默使用初始权重。"
    fail=1
fi
if ! grep -rq --exclude-dir=.git "quant_ignored_modules" FSDPTurbo/; then
    echo "ERROR: 未在 FSDPTurbo 中找到 quant_ignored_modules ——"
    echo "       训练侧排除 linear_attn 将不生效。"
    fail=1
fi
if [ "$fail" -ne 0 ]; then
    echo "install 自检失败，已中止（见 low_precision/fsdp_turbo/readme.md 版本依赖节）。"
    exit 1
fi
echo "自检通过：translate_weight_names_for_vllm / quant_ignored_modules 均在位。"

echo "7. env"
echo "请确保 PYTHONPATH 包含 FSDPTurbo: export PYTHONPATH=\$PWD/FSDPTurbo:\$PYTHONPATH"
