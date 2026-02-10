"""
PageIndex 本地化无向量 RAG 示例
================================
基于推理的检索增强生成，无需向量数据库，无需分块。

使用方法:
    python pageindex_rag_simple.py --structure <结构文件路径> --query <问题>
    
示例:
    python pageindex_rag_simple.py --structure ./results/doc_structure.json --query "文档主要内容是什么？"
"""

import json
import os
import re
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# 添加父目录到 sys.path，确保能找到 pageindex 模块
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# 导入 pageindex 工具函数
from pageindex.utils import (
    print_toc,
    structure_to_list,
    ChatGPT_API,
)

# 加载环境变量
load_dotenv()

# ============================================================================
# 模块 1: 配置管理
# ============================================================================

@dataclass
class RAGConfig:
    """RAG 配置类"""
    model: str = "qwen3-max"
    max_nodes: int = 3  # 最多选择的节点数
    context_preview_length: int = 800  # 上下文预览长度
    
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """从环境变量加载配置"""
        return cls(
            model=os.getenv("RAG_MODEL", "qwen3-max"),
            max_nodes=int(os.getenv("RAG_MAX_NODES", "3")),
        )


def validate_environment() -> bool:
    """验证环境变量配置"""
    api_key = os.getenv("CHATGPT_API_KEY")
    base_url = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    if api_key:
        print(f"[配置] API Key 已配置 (前8位): {api_key[:8]}...")
        print(f"[配置] Base URL: {base_url}")
        return True
    else:
        print("[错误] 未找到 CHATGPT_API_KEY 环境变量，请在 .env 文件中配置")
        return False


# ============================================================================
# 模块 2: LLM 调用
# ============================================================================

def call_llm(prompt: str, model: str = "qwen3-max") -> str:
    """
    调用 LLM 生成回复
    
    Args:
        prompt: 提示词
        model: 模型名称
        
    Returns:
        LLM 生成的回复文本
    """
    return ChatGPT_API(model=model, prompt=prompt)


# ============================================================================
# 模块 3: 文档结构加载
# ============================================================================

def load_structure(structure_path: str) -> Optional[Dict]:
    """
    加载本地生成的 PageIndex 树结构
    
    Args:
        structure_path: 结构文件路径 (JSON 格式)
        
    Returns:
        树结构字典，加载失败返回 None
    """
    if not os.path.exists(structure_path):
        print(f"[错误] 结构文件不存在: {structure_path}")
        return None
    
    try:
        with open(structure_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tree = data.get('structure')
        if tree:
            title = tree.get('title', '未知') if isinstance(tree, dict) else '文档集合'
            print(f"[加载] 成功加载结构文件: {structure_path}")
            print(f"[加载] 文档标题: {title}")
            return tree
        else:
            print("[错误] 结构文件中未找到 'structure' 字段")
            return None
    except json.JSONDecodeError as e:
        print(f"[错误] JSON 解析失败: {e}")
        return None


def create_node_mapping(tree: Dict) -> Dict[str, Dict]:
    """
    将树结构转换为 node_id -> node 的字典映射
    
    Args:
        tree: 树结构
        
    Returns:
        节点映射字典 {node_id: node}
    """
    node_list = structure_to_list(tree)
    return {node['node_id']: node for node in node_list if 'node_id' in node}


def print_tree_structure(tree: Dict, node_map: Dict[str, Dict]) -> None:
    """
    打印文档树结构
    
    Args:
        tree: 树结构
        node_map: 节点映射
    """
    print("\n=== 文档目录结构 ===")
    if isinstance(tree, dict):
        print_toc([tree])
    elif isinstance(tree, list):
        print_toc(tree)
    
    print(f"\n=== 共 {len(node_map)} 个节点 ===")
    for node_id, node in node_map.items():
        summary = node.get('summary', '无摘要')
        summary_preview = summary[:60] + '...' if len(summary) > 60 else summary
        print(f"  {node_id}: {node['title']} - {summary_preview}")


# ============================================================================
# 模块 4: 树搜索检索
# ============================================================================

def tree_search(
    query: str, 
    node_map: Dict[str, Dict], 
    config: RAGConfig
) -> List[str]:
    """
    使用 LLM 进行树搜索，找出与问题相关的节点
    
    Args:
        query: 用户问题
        node_map: 节点映射
        config: RAG 配置
        
    Returns:
        选中的节点 ID 列表
    """
    # 单节点文档直接返回
    if len(node_map) == 1:
        node_id = list(node_map.keys())[0]
        print(f"[检索] 文档只有一个节点，直接使用: {node_id}")
        return [node_id]
    
    # 构建节点摘要信息
    node_summaries = []
    for node_id, node in node_map.items():
        summary = node.get('summary', node.get('title', '无摘要'))
        node_summaries.append(f"- {node_id}: {node['title']} - {summary[:200]}")
    
    search_prompt = f"""你是一个文档检索助手。根据问题，从以下节点中找出最可能包含答案的节点。

问题: {query}

可用节点:
{chr(10).join(node_summaries)}

请只返回一个 JSON 数组，包含最相关的 node_id（最多{config.max_nodes}个），格式如: ["0001", "0003"]
不要返回其他内容。
"""
    
    print("[检索] 正在调用 LLM 进行树搜索...")
    response = call_llm(search_prompt, model=config.model)
    print(f"[检索] LLM 返回: {response}")
    
    # 解析返回的节点 ID
    selected_ids = _parse_node_ids(response, node_map)
    
    print(f"\n[检索] 选中的节点: {selected_ids}")
    for nid in selected_ids:
        print(f"  - {nid}: {node_map[nid]['title']}")
    
    return selected_ids


def _parse_node_ids(response: str, node_map: Dict[str, Dict]) -> List[str]:
    """
    解析 LLM 返回的节点 ID
    
    Args:
        response: LLM 返回的文本
        node_map: 节点映射
        
    Returns:
        有效的节点 ID 列表
    """
    default_ids = [list(node_map.keys())[1] if len(node_map) > 1 else list(node_map.keys())[0]]
    
    try:
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            parsed_ids = json.loads(match.group())
            valid_ids = [nid for nid in parsed_ids if nid in node_map]
            if valid_ids:
                return valid_ids
            print(f"[警告] LLM 返回的节点无效，使用默认节点")
        else:
            print(f"[警告] 无法解析 LLM 返回，使用默认节点")
    except Exception as e:
        print(f"[警告] 解析错误: {e}，使用默认节点")
    
    return default_ids


# ============================================================================
# 模块 5: 上下文提取
# ============================================================================

def extract_context(
    selected_node_ids: List[str], 
    node_map: Dict[str, Dict]
) -> str:
    """
    从选中节点提取上下文内容
    
    Args:
        selected_node_ids: 选中的节点 ID 列表
        node_map: 节点映射
        
    Returns:
        合并后的上下文文本
    """
    context_parts = []
    
    for nid in selected_node_ids:
        node = node_map[nid]
        if 'text' in node and node['text']:
            context_parts.append(f"【{node['title']}】\n{node['text']}")
        elif 'summary' in node:
            context_parts.append(f"【{node['title']}】\n{node['summary']}")
        else:
            context_parts.append(f"【{node['title']}】\n(无内容)")
    
    if context_parts:
        return "\n\n---\n\n".join(context_parts)
    return "未找到相关内容"


def print_context_preview(context: str, max_length: int = 800) -> None:
    """
    打印上下文预览
    
    Args:
        context: 上下文文本
        max_length: 最大显示长度
    """
    print(f"\n=== 检索到的上下文 (前{max_length}字) ===\n")
    print(context[:max_length])
    if len(context) > max_length:
        print(f"\n... (共 {len(context)} 字)")


# ============================================================================
# 模块 6: 答案生成
# ============================================================================

def generate_answer(
    query: str, 
    context: str, 
    config: RAGConfig
) -> str:
    """
    基于检索到的上下文生成答案
    
    Args:
        query: 用户问题
        context: 上下文内容
        config: RAG 配置
        
    Returns:
        生成的答案
    """
    answer_prompt = f"""根据以下上下文回答问题。

问题: {query}

上下文: 
{context}

请基于上下文内容，给出清晰、准确的回答。如果上下文中没有相关信息，请说明。
"""
    
    print("\n=== 正在生成答案... ===\n")
    answer = call_llm(answer_prompt, model=config.model)
    return answer


# ============================================================================
# 模块 7: RAG Pipeline 主流程
# ============================================================================

@dataclass
class RAGResult:
    """RAG 结果"""
    query: str
    selected_nodes: List[str]
    context: str
    answer: str
    success: bool = True
    error: Optional[str] = None


def run_rag_pipeline(
    structure_path: str, 
    query: str, 
    config: Optional[RAGConfig] = None,
    verbose: bool = True
) -> RAGResult:
    """
    运行完整的 RAG Pipeline
    
    Args:
        structure_path: 结构文件路径
        query: 用户问题
        config: RAG 配置 (可选)
        verbose: 是否打印详细信息
        
    Returns:
        RAGResult 结果对象
    """
    if config is None:
        config = RAGConfig()
    
    print("=" * 60)
    print("PageIndex 本地化无向量 RAG")
    print("=" * 60)
    
    # Step 0: 验证环境
    if not validate_environment():
        return RAGResult(
            query=query,
            selected_nodes=[],
            context="",
            answer="",
            success=False,
            error="环境变量配置错误"
        )
    
    # Step 1: 加载文档结构
    print(f"\n[Step 1] 加载文档结构")
    tree = load_structure(structure_path)
    if tree is None:
        return RAGResult(
            query=query,
            selected_nodes=[],
            context="",
            answer="",
            success=False,
            error="无法加载结构文件"
        )
    
    node_map = create_node_mapping(tree)
    if verbose:
        print_tree_structure(tree, node_map)
    
    # Step 2: 树搜索检索
    print(f"\n[Step 2] 树搜索检索")
    print(f"[问题] {query}")
    selected_node_ids = tree_search(query, node_map, config)
    
    # Step 3: 提取上下文
    print(f"\n[Step 3] 提取上下文")
    context = extract_context(selected_node_ids, node_map)
    if verbose:
        print_context_preview(context, config.context_preview_length)
    
    # Step 4: 生成答案
    print(f"\n[Step 4] 生成答案")
    answer = generate_answer(query, context, config)
    
    print("\n=== 生成的答案 ===\n")
    print(answer)
    print("\n" + "=" * 60)
    
    return RAGResult(
        query=query,
        selected_nodes=selected_node_ids,
        context=context,
        answer=answer,
        success=True
    )


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PageIndex 本地化无向量 RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python pageindex_rag_simple.py --structure ./results/doc_structure.json --query "文档主要内容是什么？"
  python pageindex_rag_simple.py -s ./results/doc.json -q "有哪些结论？" --model qwen3-max
        """
    )
    
    parser.add_argument(
        "-s", "--structure",
        type=str,
        default=r"D:\github_dir\pageindex_project\results\汇总文档v1_structure.json",
        help="结构文件路径 (JSON 格式)"
    )
    
    parser.add_argument(
        "-q", "--query",
        type=str,
        default="犯罪嫌疑人涉嫌什么罪名？具体犯罪事实是什么？",
        help="用户问题"
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="qwen3-max",
        help="LLM 模型名称 (默认: qwen3-max)"
    )
    
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=3,
        help="最多选择的节点数 (默认: 3)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="打印详细信息"
    )
    
    args = parser.parse_args()
    
    config = RAGConfig(
        model=args.model,
        max_nodes=args.max_nodes
    )
    
    result = run_rag_pipeline(
        structure_path=args.structure,
        query=args.query,
        config=config,
        verbose=args.verbose
    )
    
    if not result.success:
        print(f"\n[失败] {result.error}")
        exit(1)


if __name__ == "__main__":
    main()
