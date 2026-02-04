<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=1280, height=720">
    <title>Megatron CP 实现原理</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            width: 1280px;
            height: 720px;
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #333;
            overflow: hidden;
        }
        .container {
            width: 100%;
            height: 100%;
            padding: 40px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: auto 1fr 1fr;
            gap: 20px;
        }
        .header {
            grid-column: 1 / -1;
            background: rgba(255,255,255,0.95);
            border-radius: 12px;
            padding: 20px 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .header h1 {
            font-size: 36px;
            color: #1e3c72;
            margin-bottom: 8px;
        }
        .header .subtitle {
            font-size: 16px;
            color: #666;
        }
        .card {
            background: rgba(255,255,255,0.95);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .card-title {
            font-size: 20px;
            color: #1e3c72;
            margin-bottom: 12px;
            border-bottom: 2px solid #2a5298;
            padding-bottom: 8px;
        }
        .flow-diagram {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .flow-step {
            background: linear-gradient(90deg, #e8f4f8 0%, #d0e8f0 100%);
            padding: 10px 15px;
            border-radius: 8px;
            border-left: 4px solid #2a5298;
            font-size: 14px;
        }
        .flow-step strong {
            color: #1e3c72;
        }
        .key-points {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .point {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 6px;
            font-size: 13px;
        }
        .point-label {
            color: #2a5298;
            font-weight: bold;
            margin-bottom: 4px;
        }
        .code-block {
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 12px;
            border-radius: 6px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            line-height: 1.6;
        }
        .code-comment {
            color: #6a9955;
        }
        .code-keyword {
            color: #569cd6;
        }
        .code-string {
            color: #ce9178;
        }
        .highlight {
            background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
            padding: 10px 15px;
            border-radius: 6px;
            font-weight: bold;
            color: #1e3c72;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🚀 Megatron-LM Context Parallelism (CP) 实现原理</h1>
            <div class="subtitle">基于 Transformer Engine Ring Attention 的超长序列并行方案</div>
        </div>

        <!-- Left Column -->
        <div class="card" style="grid-row: 2">
            <div class="card-title">📋 核心实现流程</div>
            <div class="flow-diagram">
                <div class="flow-step">
                    <strong>1. 初始化阶段</strong><br>
                    parallel_state.initialize_model_parallel() → 创建 _CONTEXT_PARALLEL_GROUP
                </div>
                <div class="flow-step">
                    <strong>2. Attention 层创建</strong><br>
                    TEDotProductAttention.__init__() → 配置 CP 通信类型 (p2p/a2a) + 创建专用 CUDA Stream
                </div>
                <div class="flow-step">
                    <strong>3. Ring Attention 前向传播</strong><br>
                    输入: [seq/cp, batch, heads, dim] → Round 0: 本地 attention → Round N: 交换 KV → 输出
                </div>
                <div class="flow-step">
                    <strong>4. 后处理</strong><br>
                    Output Projection + AllGather → 完整序列输出
                </div>
            </div>
        </div>

        <div class="card" style="grid-row: 3">
            <div class="card-title">🔑 关键技术点</div>
            <div class="key-points">
                <div class="point">
                    <div class="point-label">进程组管理</div>
                    <div style="font-size: 12px; color: #666;">parallel_state.py:972-999</div>
                </div>
                <div class="point">
                    <div class="point-label">Ring Attention</div>
                    <div style="font-size: 12px; color: #666;">P2P 环形通信 (TE 内部)</div>
                </div>
                <div class="point">
                    <div class="point-label">CP Stream</div>
                    <div style="font-size: 12px; color: #666;">通信计算重叠优化</div>
                </div>
                <div class="point">
                    <div class="point-label">动态 CP 组</div>
                    <div style="font-size: 12px; color: #666;">运行时切换 CP 配置</div>
                </div>
            </div>
            <div class="highlight" style="margin-top: 15px;">
                ⚡ 核心优势：突破单 GPU 内存限制，支持 8K-128K 超长序列训练
            </div>
        </div>

        <!-- Right Column -->
        <div class="card" style="grid-row: 2 / 4">
            <div class="card-title">💻 核心代码示例</div>
            <div class="code-block">
<span class="code-comment"># 1. 配置 CP 参数</span>
config = <span class="code-keyword">TransformerConfig</span>(
    context_parallel_size=<span class="code-string">2</span>,    <span class="code-comment"># CP = 2</span>
    cp_comm_type=<span class="code-string">"p2p"</span>,           <span class="code-comment"># Ring Attention</span>
    sequence_parallel=<span class="code-keyword">True</span>,
)

<span class="code-comment"># 2. 创建 Attention 层 (自动集成 CP)</span>
attention = <span class="code-keyword">TEDotProductAttention</span>(
    config=config,
    layer_number=<span class="code-string">1</span>,
    attn_mask_type=AttnMaskType.causal,
)

<span class="code-comment"># 3. 前向传播 - CP 通信自动处理</span>
<span class="code-keyword">context</span>, _ = attention(
    query=query,     <span class="code-comment"># [seq/2, batch, heads, dim]</span>
    key=key,
    value=value,
)

<span class="code-comment"># 4. Ring Attention 通信过程 (TE 内部)</span>
<span class="code-comment"># Round 0: Rank0 计算 [0-2047]×[0-2047]</span>
<span class="code-comment">#          Rank1 计算 [2048-4095]×[2048-4095]</span>
<span class="code-comment"># Round 1: Rank0 接收 KV(2048-4095) from Rank1</span>
<span class="code-comment">#          Rank1 接收 KV(0-2047) from Rank0</span>
<span class="code-comment"># → 完成 Ring Attention 计算</span>
            </div>
            <div style="margin-top: 15px; padding: 15px; background: #e8f4f8; border-radius: 8px;">
                <div style="font-size: 14px; color: #1e3c72; margin-bottom: 8px;">📌 快速启动命令</div>
                <div style="font-family: monospace; font-size: 12px; background: #fff; padding: 8px; border-radius: 4px;">
torchrun --nproc_per_node=8 pretrain_gpt.py \
  --tensor-model-parallel-size 4 \
  --context-parallel-size 2 \
  --cp-comm-type p2p \
  --seq-length 8192
                </div>
            </div>
        </div>
    </div>
</body>
</html>
