# Change Log
All notable changes to this project will be documented in this file.

## Beta - 2026-02-10

### Added
- [x] **卷宗文件夹处理功能** (`folder_index_main`): 支持处理包含多个 TXT 文件的卷宗文件夹
  - 每个 TXT 文件的文件名自动作为一个节点标题
  - 支持 `--max-tokens-for-subnodes` 参数控制子节点提取阈值（默认 1024 token）
  - 超过阈值的材料会自动调用 LLM 提取子结构
- [x] **TXT 文件处理功能** (`txt_index_main`): 支持单个 TXT 文件的结构提取
  - 按 token 数量虚拟分页，模拟 PDF 的 page_list 结构
  - 支持 `--tokens-per-page` 参数控制每页 token 数（默认 500）
- [x] **LLM 调用进度反馈**: 在耗时的 LLM 调用处添加时间估算和耗时统计
  - `check_title_appearance_in_start_concurrent`: 标题验证进度
  - `verify_toc`: TOC 验证进度
  - `generate_summaries_for_structure`: 摘要生成进度

### Changed
- [x] `run_pageindex_txt.py` 新增命令行参数:
  - `--folder_path`: 卷宗文件夹路径
  - `--max-tokens-for-subnodes`: 子节点提取阈值
  - `--tokens-per-page`: TXT 虚拟分页 token 数

### Files Modified
- `pageindex/page_index.py`: 新增 `folder_index_main`, `txt_index_main` 函数
- `pageindex/utils.py`: 新增 `get_txt_page_tokens` 函数，添加进度反馈
- `run_pageindex_txt.py`: 新增文件夹处理逻辑和相关参数

---

## Beta - 2025-04-23

### Fixed
- [x] Fixed a bug introduced on April 18 where `start_index` was incorrectly passed.

## Beta - 2025-04-03

### Added
- [x] Add node_id, node summary
- [x] Add document discription

### Changed
- [x] Change "child_nodes" -> "nodes" to simplify the structure
