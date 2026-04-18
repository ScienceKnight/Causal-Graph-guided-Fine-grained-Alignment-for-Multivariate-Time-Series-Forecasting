import asyncio
import nest_asyncio
nest_asyncio.apply()
import os
import json
import re
import numpy as np
import logging
from lightrag.lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = 'CasualAlign/Generate/Code'
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2.5",
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        llm_model_kwargs={
            "host": "http://localhost:11051",
            "options": {"num_ctx": 32768},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model="nomic-embed-text", host="http://localhost:11051"
            ),
        ),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)

def parse_llm_response(response_text, all_variables):
    """

    (HUFL, causes, OT, {"lag": [2], "season": ["summer"], "confidence": 0.9})
    [(src, tgt, relation, time_conditions, confidence), ...]
    """
    edges = []

    pattern = r'\(\s*([A-Za-z0-9_]+)\s*,\s*([a-z]+)\s*,\s*([A-Za-z0-9_]+)\s*,\s*(\{.*?\})\s*\)'
    matches = re.findall(pattern, response_text, re.DOTALL)
    for match in matches:
        src, rel, tgt, cond_str = match
        if src not in all_variables or tgt not in all_variables:
            continue
        try:
            cond = json.loads(cond_str)
        except:
            cond = {}
        confidence = cond.pop('confidence', 1.0)  
        edges.append((src, tgt, rel, cond, confidence))
    return edges

def generate_structured_knowledge(rag, variables):

    all_edges = []
    prompts = {}
    for var in variables:
        prompt = f"""You are a data scientist proficient in time series analysis.
The dataset contains the following variables: {variables}.
Current variable: {var}.
Task: Analyze the causal relationships between {var} and other variables. For each relationship, provide:
- Source variable (should be {var} or another variable)
- Relation type (causes, correlates, affects, etc.)
- Target variable
- Time conditions (JSON object with possible keys: lag (list of ints), season (list of strings), hour_range (list of two ints), confidence (float 0-1))

Output format: each relationship on a new line as (source, relation, target, {{"lag": [...], "season": [...], "hour_range": [...], "confidence": ...}})

Only output the relationships, no additional text.
"""
        response = rag.query(prompt, param=QueryParam(mode="naive"))
        edges = parse_llm_response(response, variables)
        all_edges.extend(edges)
        prompts[var] = f"Variable: {var}\nDescription: {response}"
    return all_edges, prompts

def build_adjacency_and_metadata(edges, variables, output_adj_path, output_edges_json):
    """
    JSON
    """
    N = len(variables)
    var_to_idx = {v: i for i, v in enumerate(variables)}
    adj = np.zeros((N, N))
    edge_list = []
    for src, tgt, rel, cond, conf in edges:
        i, j = var_to_idx[src], var_to_idx[tgt]
        adj[i, j] = conf  
        edge_list.append({
            'source': src,
            'target': tgt,
            'relation': rel,
            'time_conditions': cond,
            'confidence': conf
        })
    np.save(output_adj_path, adj)
    with open(output_edges_json, 'w') as f:
        json.dump(edge_list, f, indent=2)
    return adj, edge_list

def MKG(txt_path, variables, output_prompt_file, output_adj_file, output_edges_json):
    rag = asyncio.run(initialize_rag())
    with open(txt_path, "r", encoding="utf-8") as f:
        rag.insert(f.read())
    all_edges, prompts = generate_structured_knowledge(rag, variables)
    with open(output_prompt_file, "w", encoding="utf-8") as f:
        for var, prompt in prompts.items():
            f.write(f"{var}:\n{prompt}\n\n")
    build_adjacency_and_metadata(all_edges, variables, output_adj_file, output_edges_json)

    print(f"MKG constructed: prompts saved to {output_prompt_file}, graph saved to {output_adj_file}, edges metadata saved to {output_edges_json}")

def response_rag(prompt, txt_path):
    rag = asyncio.run(initialize_rag())
    with open(txt_path, "r", encoding="utf-8") as f:
        rag.insert(f.read())
    response = rag.query(prompt, param=QueryParam(mode="naive"))
    return response

if __name__ == "__main__":
    txt_path = "./dataset_txt/ETT.txt"       
    variables = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    output_prompt_file = "./prompts/ETT_prompt.txt"
    output_adj_file = "./graphs/ETT_graph.npy"
    output_edges_json = "./graphs/ETT_edges.json"
    MKG(txt_path, variables, output_prompt_file, output_adj_file, output_edges_json)