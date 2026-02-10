import json
import os
from datetime import datetime
from pageindex.utils import print_toc, remove_fields, structure_to_list, ChatGPT_API

# 配置模型
MODEL = "qwen3-max"

# 创建日志目录
LOG_DIR = r"D:\github_dir\pageindex_project\logs\query_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 日志文件路径
log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"query_log_{log_timestamp}.txt")

# 封装 LLM 调用
def call_llm(prompt, model=MODEL):
    return ChatGPT_API(model=model, prompt=prompt)

# 创建节点映射的辅助函数
def create_node_mapping(tree):
    """将树结构转换为 node_id -> node 的字典映射"""
    node_list = structure_to_list(tree)
    return {node['node_id']: node for node in node_list if 'node_id' in node}

# 加载生成的结构
strcture_path=r"D:\github_dir\pageindex_project\results\汇总文档v1_structure.json"
with open(strcture_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

tree = data['structure']

# 查看树结构
print("=== 文档结构 ===")
print_toc(tree)

# 创建节点映射 (node_id -> node)
node_map = create_node_mapping(tree)
print(f"\n=== 节点数量: {len(node_map)} ===")
for node_id, node in node_map.items():
    print(f"  {node_id}: {node['title']}")

# 检查是否有 summary 可用于检索
print("\n=== 节点摘要 ===")
for node_id, node in node_map.items():
    summary = node.get('summary', '无摘要')
    print(f"  {node_id}: {summary[:100]}..." if len(summary) > 100 else f"  {node_id}: {summary}")

# 示例问答
query = "犯罪嫌疑人涉嫌什么罪名？具体犯罪事实是什么？"
print(f"\n=== 问题: {query} ===")

# Step 1: 树搜索 - 让 LLM 根据 summary 找相关节点
# 对于只有一个节点的简单文档，直接使用该节点
if len(node_map) == 1:
    target_node_id = list(node_map.keys())[0]
    print(f"文档只有一个节点，直接使用: {target_node_id}")
else:
    # 多节点时使用 LLM 搜索
    # 构建节点摘要信息供 LLM 选择
    node_summaries = []
    for node_id, node in node_map.items():
        summary = node.get('summary', node.get('title', '无摘要'))
        node_summaries.append(f"- {node_id}: {node['title']} - {summary[:200]}")
    
    search_prompt = f"""你是一个文档检索助手。根据问题，从以下节点中找出最可能包含答案的节点。

问题: {query}

可用节点:
{chr(10).join(node_summaries)}

请只返回一个 JSON 数组，包含最相关的 node_id（最多3个），格式如: ["0005", "0013"]
不要返回其他内容。
"""
    node_ids_response = call_llm(search_prompt)
    print(f"LLM 返回的节点: {node_ids_response}")
    
    # 解析 LLM 返回的节点 ID
    try:
        # 尝试从返回内容中提取 JSON 数组
        import re
        match = re.search(r'\[.*?\]', node_ids_response, re.DOTALL)
        if match:
            selected_ids = json.loads(match.group())
            # 过滤出有效的节点 ID
            valid_ids = [nid for nid in selected_ids if nid in node_map]
            if valid_ids:
                target_node_id = valid_ids[0]  # 使用第一个有效节点
                print(f"选中节点: {target_node_id}")
            else:
                target_node_id = list(node_map.keys())[1] if len(node_map) > 1 else list(node_map.keys())[0]
                print(f"LLM 返回的节点无效，使用默认节点: {target_node_id}")
        else:
            target_node_id = list(node_map.keys())[1] if len(node_map) > 1 else list(node_map.keys())[0]
            print(f"无法解析 LLM 返回，使用默认节点: {target_node_id}")
    except Exception as e:
        print(f"解析错误: {e}")
        target_node_id = list(node_map.keys())[1] if len(node_map) > 1 else list(node_map.keys())[0]

# Step 2: 收集所有选中节点的内容作为上下文
# 如果有多个相关节点，合并它们的内容
if 'valid_ids' in dir() and len(valid_ids) > 1:
    print(f"\n合并 {len(valid_ids)} 个节点的内容...")
    context_parts = []
    for nid in valid_ids:
        node = node_map[nid]
        if 'text' in node:
            context_parts.append(f"【{node['title']}】\n{node['text']}")
        elif 'summary' in node:
            context_parts.append(f"【{node['title']}】\n{node['summary']}")
    context = "\n\n---\n\n".join(context_parts)
else:
    # 单节点情况
    target_node = node_map[target_node_id]
    if 'text' not in target_node:
        print("\n注意: 节点没有文本内容。需要在生成时设置 --if-add-node-text=yes")
        print("可以使用 summary 作为上下文:")
        context = target_node.get('summary', '无内容')
    else:
        context = target_node['text']

print(f"\n=== 上下文 (前500字) ===\n{context[:500]}...")

# Step 3: 生成答案
answer_prompt = f"""根据以下上下文回答问题。

问题: {query}

上下文: 
{context}

请直接回答问题。
"""

print("\n=== 正在生成答案... ===")

# 保存完整的请求到日志文件
with open(log_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write(f"查询日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("【用户问题】\n")
    f.write(f"{query}\n\n")
    
    f.write("【选中的节点】\n")
    if 'valid_ids' in dir() and valid_ids:
        f.write(f"节点 ID: {', '.join(valid_ids)}\n")
        for nid in valid_ids:
            f.write(f"  - {nid}: {node_map[nid]['title']}\n")
    else:
        f.write(f"节点 ID: {target_node_id}\n")
        f.write(f"  - {target_node_id}: {node_map[target_node_id]['title']}\n")
    f.write("\n")
    
    f.write("【上下文内容】\n")
    f.write("-" * 80 + "\n")
    f.write(context)
    f.write("\n")
    f.write("-" * 80 + "\n\n")
    
    f.write("【完整的 LLM 请求提示词】\n")
    f.write("-" * 80 + "\n")
    f.write(answer_prompt)
    f.write("\n")
    f.write("-" * 80 + "\n\n")

print(f"✓ 日志已保存到: {log_file}")

answer = call_llm(answer_prompt)
print(f"\n=== 答案 ===\n{answer}")

# 将答案也追加到日志文件
with open(log_file, 'a', encoding='utf-8') as f:
    f.write("【LLM 返回的答案】\n")
    f.write("-" * 80 + "\n")
    f.write(answer)
    f.write("\n")
    f.write("-" * 80 + "\n")