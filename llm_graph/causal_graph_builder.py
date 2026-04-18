import json
import torch
import numpy as np
from typing import List, Dict

class CausalGraphBuilder:

    def __init__(self, llm_client=None, var_names: List[str] = None, domain_desc: str = None):
        self.llm_client = llm_client
        self.var_names = var_names
        self.domain_desc = domain_desc
        self.N = len(var_names) if var_names else 0

    def build_static_meta_graph(self, save_path: str = None) -> torch.Tensor:

        prompt = self._build_prompt()
        
        if self.llm_client is not None:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            causal_relations = self._parse_llm_response(response.choices[0].message.content)
        else:
            causal_relations = self._mock_llm_response()
        
        adj = np.zeros((self.N, self.N), dtype=np.float32)
        var_to_idx = {name: i for i, name in enumerate(self.var_names)}
        
        for (src, dst) in causal_relations:
            if src in var_to_idx and dst in var_to_idx:
                adj[var_to_idx[src], var_to_idx[dst]] = 1.0
        
        adj = adj + np.eye(self.N) 
        adj = torch.tensor(adj, dtype=torch.float32)
        
        if save_path:
            torch.save(adj, save_path)
            print(f"{save_path}")
        
        return adj

    def _build_prompt(self) -> str:
        var_list = "\n".join([f"- {name}" for name in self.var_names])
        prompt = f"""
	prompt template
        """
        return prompt.strip()

    def _parse_llm_response(self, response: str) -> List[tuple]:
        try:
            relations = json.loads(response)
            return [(rel[0], rel[1]) for rel in relations]
        except:
            print("LLM")
            return self._mock_llm_response()

 def _mock_llm_response(self) -> List[tuple]:
        return [
            ("HUFL", "OT"), ("HULL", "OT"),
            ("MUFL", "OT"), ("MULL", "OT"),
            ("LUFL", "OT"), ("LULL", "OT"),
            ("HUFL", "HULL"), ("MUFL", "MULL"),
            ("LUFL", "LULL")
        ]