import argparse
import os
import json
import time
from pageindex import *
from pageindex.page_index_md import md_to_tree

if __name__ == "__main__":
    # Set up argument parser
    folder_path="D:\github_dir\pageindex_project\三人八起"
   
   
    parser = argparse.ArgumentParser(description='Process PDF, TXT or Markdown document and generate structure')
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--txt_path', type=str, help='Path to the TXT file')
    parser.add_argument('--folder_path', type=str, help='Path to a folder containing TXT files (case folder)',default=folder_path)
    parser.add_argument('--md_path', type=str, help='Path to the Markdown file')

    parser.add_argument('--model', type=str, default='qwen3-max', help='Model to use')

    parser.add_argument('--toc-check-pages', type=int, default=20, 
                      help='Number of pages to check for table of contents (PDF only)')
    parser.add_argument('--max-pages-per-node', type=int, default=10,
                      help='Maximum number of pages per node (PDF only)')
    parser.add_argument('--max-tokens-per-node', type=int, default=20000,
                      help='Maximum number of tokens per node (PDF only)')
    
    # TXT/Folder specific arguments
    parser.add_argument('--tokens-per-page', type=int, default=500,
                      help='Target tokens per virtual page for TXT files (TXT only)')
    parser.add_argument('--max-tokens-for-subnodes', type=int, default=1024,
                      help='Max tokens threshold for extracting subnodes (Folder only)')

    parser.add_argument('--if-add-node-id', type=str, default='yes',
                      help='Whether to add node id to the node')
    parser.add_argument('--if-add-node-summary', type=str, default='yes',
                      help='Whether to add summary to the node')
    parser.add_argument('--if-add-doc-description', type=str, default='no',
                      help='Whether to add doc description to the doc')
    parser.add_argument('--if-add-node-text', type=str, default='yes',
                      help='Whether to add text to the node')
                      
    # Markdown specific arguments
    parser.add_argument('--if-thinning', type=str, default='no',
                      help='Whether to apply tree thinning for markdown (markdown only)')
    parser.add_argument('--thinning-threshold', type=int, default=5000,
                      help='Minimum token threshold for thinning (markdown only)')
    parser.add_argument('--summary-token-threshold', type=int, default=200,
                      help='Token threshold for generating summaries (markdown only)')
    args = parser.parse_args()
    
    # Validate that exactly one file type is specified
    if args.pdf_path and args.md_path:
        raise ValueError("Only one of --pdf_path or --md_path can be specified")
    
    # Folder takes highest priority if specified
    if args.folder_path:
        # Validate folder
        if not os.path.isdir(args.folder_path):
            raise ValueError(f"Folder not found: {args.folder_path}")
            
        # Configure options
        opt = config(
            model=args.model,
            toc_check_page_num=args.toc_check_pages,
            max_page_num_each_node=args.max_pages_per_node,
            max_token_num_each_node=args.max_tokens_per_node,
            if_add_node_id=args.if_add_node_id,
            if_add_node_summary=args.if_add_node_summary,
            if_add_doc_description=args.if_add_doc_description,
            if_add_node_text=args.if_add_node_text
        )

        # Process the folder
        print(f'[Folder] 开始解析卷宗: {args.folder_path}')
        start_time = time.time()
        
        from pageindex.page_index import folder_index_main
        toc_with_page_number = folder_index_main(
            args.folder_path, 
            opt, 
            tokens_per_page=args.tokens_per_page,
            max_tokens_for_subnodes=args.max_tokens_for_subnodes
        )
        
        elapsed_time = time.time() - start_time
        print(f'[Folder] 解析完成, 耗时: {elapsed_time:.2f} 秒')
        
        # Save results
        folder_name = os.path.basename(args.folder_path.rstrip('/\\'))
        output_dir = './results'
        output_file = f'{output_dir}/{folder_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)
        
        print(f'Tree structure saved to: {output_file}')
    
    # TXT takes priority if specified
    elif args.txt_path:
        # Validate TXT file
        if not args.txt_path.lower().endswith('.txt'):
            raise ValueError("TXT file must have .txt extension")
        if not os.path.isfile(args.txt_path):
            raise ValueError(f"TXT file not found: {args.txt_path}")
            
        # Configure options
        opt = config(
            model=args.model,
            toc_check_page_num=args.toc_check_pages,
            max_page_num_each_node=args.max_pages_per_node,
            max_token_num_each_node=args.max_tokens_per_node,
            if_add_node_id=args.if_add_node_id,
            if_add_node_summary=args.if_add_node_summary,
            if_add_doc_description=args.if_add_doc_description,
            if_add_node_text=args.if_add_node_text
        )

        # Process the TXT
        print(f'[TXT] 开始解析: {args.txt_path}')
        start_time = time.time()
        
        from pageindex.page_index import txt_index_main
        toc_with_page_number = txt_index_main(args.txt_path, opt, tokens_per_page=args.tokens_per_page)
        
        elapsed_time = time.time() - start_time
        print(f'[TXT] 解析完成, 耗时: {elapsed_time:.2f} 秒')
        
        # Save results
        txt_name = os.path.splitext(os.path.basename(args.txt_path))[0]    
        output_dir = './results'
        output_file = f'{output_dir}/{txt_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)
        
        print(f'Tree structure saved to: {output_file}')
            
    elif args.pdf_path:
        # Validate PDF file
        if not args.pdf_path.lower().endswith('.pdf'):
            raise ValueError("PDF file must have .pdf extension")
        if not os.path.isfile(args.pdf_path):
            raise ValueError(f"PDF file not found: {args.pdf_path}")
            
        # Configure options
        opt = config(
            model=args.model,
            toc_check_page_num=args.toc_check_pages,
            max_page_num_each_node=args.max_pages_per_node,
            max_token_num_each_node=args.max_tokens_per_node,
            if_add_node_id=args.if_add_node_id,
            if_add_node_summary=args.if_add_node_summary,
            if_add_doc_description=args.if_add_doc_description,
            if_add_node_text=args.if_add_node_text
        )

        # Process the PDF
        print(f'[PDF] 开始解析: {args.pdf_path}')
        start_time = time.time()
        
        toc_with_page_number = page_index_main(args.pdf_path, opt)
        
        elapsed_time = time.time() - start_time
        print(f'[PDF] 解析完成, 耗时: {elapsed_time:.2f} 秒')
        
        # Save results
        pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]    
        output_dir = './results'
        output_file = f'{output_dir}/{pdf_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)
        
        print(f'Tree structure saved to: {output_file}')
            
    elif args.md_path:
        # Validate Markdown file
        if not args.md_path.lower().endswith(('.md', '.markdown')):
            raise ValueError("Markdown file must have .md or .markdown extension")
        if not os.path.isfile(args.md_path):
            raise ValueError(f"Markdown file not found: {args.md_path}")
            
        # Process markdown file
        print('Processing markdown file...')
        
        # Process the markdown
        import asyncio
        
        print(f'[Markdown] 开始解析: {args.md_path}')
        start_time = time.time()
        
        # Use ConfigLoader to get consistent defaults (matching PDF behavior)
        from pageindex.utils import ConfigLoader
        config_loader = ConfigLoader()
        
        # Create options dict with user args
        user_opt = {
            'model': args.model,
            'if_add_node_summary': args.if_add_node_summary,
            'if_add_doc_description': args.if_add_doc_description,
            'if_add_node_text': args.if_add_node_text,
            'if_add_node_id': args.if_add_node_id
        }
        
        # Load config with defaults from config.yaml
        opt = config_loader.load(user_opt)
        
        toc_with_page_number = asyncio.run(md_to_tree(
            md_path=args.md_path,
            if_thinning=args.if_thinning.lower() == 'yes',
            min_token_threshold=args.thinning_threshold,
            if_add_node_summary=opt.if_add_node_summary,
            summary_token_threshold=args.summary_token_threshold,
            model=opt.model,
            if_add_doc_description=opt.if_add_doc_description,
            if_add_node_text=opt.if_add_node_text,
            if_add_node_id=opt.if_add_node_id
        ))
        
        elapsed_time = time.time() - start_time
        print(f'[Markdown] 解析完成, 耗时: {elapsed_time:.2f} 秒')
        
        # Save results
        md_name = os.path.splitext(os.path.basename(args.md_path))[0]    
        output_dir = './results'
        output_file = f'{output_dir}/{md_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)
        
        print(f'Tree structure saved to: {output_file}')