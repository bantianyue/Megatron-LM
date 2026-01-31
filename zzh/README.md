# Megatron-LM 分析文档

本目录包含 Megatron-LM 的架构分析和并行策略文档，定期更新并推送到 GitHub 仓库的 **main** 分支。

## 🔄 分支说明

**默认分支**: `main` ⭐

**重要**: 以后所有文档更新都推送到 `main` 分支，而不是 `master` 分支。

---

## 📁 文档列表

### 1. MEGATRON_ANALYSIS.md
- **内容**: Megatron-LM 架构分析（中文）
- **包含**:
  - 核心架构详解
  - MCore 目录结构完整说明（新增所有子模块）
  - 并行策略详解（TP/PP/DP/SP/CP）
  - 训练循环架构
  - 模型实现详解
  - 内存优化技术
  - 性能优化

### 2. TP_CP_EP_PARALLELISM_ANALYSIS.md ⭐
- **内容**: TP/CP/EP 并行策略深度分析
- **包含**:
  - 张量并行 (TP) 实现和接口
  - 上下文并行 (CP) 实现和接口
  - 专家并行 (EP) 实现和接口
  - 完整代码示例
  - API 速查表

### 3. CUDA_DEPENDENCY_ANALYSIS.md
- **内容**: CUDA Toolkit 依赖深度分析
- **包含**:
  - 动态编译的 CUDA 扩展
  - 第三方 CUDA 库依赖
  - CUDA 版本要求
  - 安装与配置指南

### 4. TRAINING_MODULE_ANALYSIS.md
- **内容**: Training 模块完整分析
- **包含**:
  - megatron/training/ 目录结构
  - 核心模块功能
  - datasets/ 和 tokenizer/ 详解

### 5. design_philosophy.md
- **内容**: 架构图设计哲学

### 6. megatron_architecture.pdf
- **内容**: Megatron-LM 架构图（带 MCore 标识）

---

## 🔄 同步规则

**重要**: 每次更新文档后，务必同步到 `main` 分支！

### 同步步骤

```bash
# 1. 确保当前在 main 分支
git checkout main

# 2. 复制文件到 zzh 目录
cd D:\02_source\Megatron-LM
cp MEGATRON_ANALYSIS.md TP_CP_EP_PARALLELISM_ANALYSIS.md \
   CUDA_DEPENDENCY_ANALYSIS.md TRAINING_MODULE_ANALYSIS.md \
   megatron_architecture.pdf zzh/

# 3. 提交到 Git
git add zzh/
git commit -m "docs: 更新文档说明"

# 4. 推送到远程（main 分支）
git push
```

---

## 🌐 远程仓库

- **仓库地址**: https://github.com/bantianyue/Megatron-LM
- **分支**: `main` ⭐（默认分支）
- **目录**: `zzh/`

---

## 📝 更新记录

| 日期 | 更新内容 | Commit |
|------|---------|--------|
| 2025-01-31 10:00 | 切换到 main 分支 | - |
| 2025-01-30 16:20 | 添加 TP/CP/EP 并行策略分析 | a63fffd |
| 2025-01-30 16:20 | 添加 Training 模块分析 | 42aa048 |
| 2025-01-30 16:20 | 添加 zzh 目录说明 | a8bd90d |
| 2025-01-30 16:20 | 添加 CUDA 依赖分析 | 39fcd65 |
| 2025-01-30 16:20 | 完善 MCore 目录结构 | 4765336 |
| 2025-01-30 14:57 | 添加架构分析文档 | 6c42704 |

---

## 🏷️ 文档标识

- **🏆 MCore** = Megatron Core 组件，位于 `megatron/core/` 目录
- **未标注** = Training Framework 组件，位于 `megatron/training/` 目录

---

## 📌 快速链接

- [在线查看](https://github.com/bantianyue/Megatron-LM/tree/main/zzh)
- [Megatron-LM 原仓库](https://github.com/NVIDIA/Megatron-LM)

---

*本文档由 Claude Code 生成并维护*
*最后更新: 2025-01-31*
*默认分支: main* ⭐
