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
# 版本依赖：需含 quant_ignored_modules 支持（训练侧排除 linear_attn 依赖此项）及
# EFSDP 条件修正、LPAG×MoE×EP=1 防御等修正。相关修改将直接合入 FSDPTurbo 主仓；
# 当前 pin b1793de 为已验证基线，合入后请更新 pin 并补记 commit 号。
git clone https://gitcode.com/Ascend/FSDPTurbo.git
cd FSDPTurbo
git checkout b1793de
pip install -e .
cd ..

echo "4. install verl"
# 版本依赖：需含 quant_ignored_modules 配置支持及 vLLM 量化权重名映射
# （translate_weight_names_for_vllm）、FSDP 模型合并器 mesh/placement 兼容等修正。
# 相关修改将直接合入 verl 主仓；当前 pin 55a06a16 为已验证基线，合入后请更新 pin 并补记 commit 号。
git clone https://github.com/verl-project/verl
cd verl
git checkout 55a06a16
sed -i '/^triton-ascend==/ s/^/# /' requirements-npu.txt
pip install -r requirements-npu.txt
cd ..

echo "5. install triton-ascend && transformers"
pip install transformers==5.10.4
pip install triton-ascend==3.2.2 --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn

echo "6. check quant_ignored_modules support"
# 轻量自检：mxfp8 launcher 依赖 quant_ignored_modules（训练侧排除 linear_attn）。
# 缺失仅告警不中断（bf16 链路不需要该项）。
if ! grep -rq --exclude-dir=.git "quant_ignored_modules" verl/; then
    echo "WARN: 未在 verl 中找到 quant_ignored_modules 支持——MXFP8 训练侧排除 linear_attn 将不生效（bf16 链不受影响），请升级 verl 至含该支持的版本。"
fi
if ! grep -rq --exclude-dir=.git "quant_ignored_modules" FSDPTurbo/; then
    echo "WARN: 未在 FSDPTurbo 中找到 quant_ignored_modules 支持——MXFP8 训练侧排除 linear_attn 将不生效（bf16 链不受影响），请升级 FSDPTurbo 至含该支持的版本。"
fi

echo "7. env"
echo "请确保 PYTHONPATH 包含 FSDPTurbo: export PYTHONPATH=\$PWD/FSDPTurbo:\$PYTHONPATH"
