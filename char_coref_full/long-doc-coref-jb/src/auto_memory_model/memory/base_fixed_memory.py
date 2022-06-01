import torch
from pytorch_utils.modules import MLP
from auto_memory_model.memory import BaseMemory


class BaseFixedMemory(BaseMemory):
    def __init__(self, num_cells=10, **kwargs):
        super(BaseFixedMemory, self).__init__(**kwargs)

        self.num_cells = num_cells
        # Fixed memory cells need to predict fertility of memory and mentions
        self.fert_mlp = MLP(input_size=self.mem_size + self.num_feats * self.emb_size,
                            hidden_size=self.mlp_size, output_size=1, num_hidden_layers=self.mlp_depth,
                            bias=True, drop_module=self.drop_module)

    def initialize_memory(self):
        """Initialize the memory to null."""
        mem = torch.zeros(self.num_cells, self.mem_size).to(self.device)
        ent_counter = torch.tensor([0 for i in range(self.num_cells)], requires_grad=False).to(self.device)
        last_mention_idx = torch.tensor([0 for i in range(self.num_cells)], requires_grad=False).to(self.device)
        return mem, ent_counter, last_mention_idx


    def interpret_coref_new_score(self, coref_new_scores):
        pred_max_idx = torch.argmax(coref_new_scores).item()
        coref_new_scores1 = torch.cat([coref_new_scores[0:pred_max_idx], coref_new_scores[pred_max_idx+1:]])
        pred_max2_idx = torch.argmax(coref_new_scores1).item()
        coref_new_scores2 = torch.cat([coref_new_scores1[0:pred_max2_idx], coref_new_scores1[pred_max2_idx+1:]])
        pred_max3_idx = torch.argmax(coref_new_scores2).item()
        
        if (coref_new_scores2[pred_max3_idx] > (0.2* coref_new_scores[pred_max_idx])) & (coref_new_scores1[pred_max2_idx] > (0.8* coref_new_scores[pred_max_idx])):
            return -1, 'i'
        if pred_max_idx < self.num_cells:
            # Coref
            return pred_max_idx, 'c'
        elif pred_max_idx == self.num_cells:
            # Overwrite/No Space/Ignore
            return -1, None
