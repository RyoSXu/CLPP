# datasets/crystal_phdos_dataset.py
import os
import torch
import matgl
from pymatgen.core import Structure
from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MGLDataset
from tqdm import tqdm
import warnings

class CrystalPhDOSDataset(MGLDataset):
    """
    数据集：每个样本包含晶体结构和对应的声子态密度
    structure: ./data/mp-XXXX.cif
    phdos: ./data/tgtdos.csv (mp-id, 51 values)
    """

    def __init__(self, csv_path="./data/tgtdos.csv", data_dir="./data", element_types=None, threebody_cutoff=6.0):
        self.csv_path = csv_path
        self.structure_dir = data_dir

        # 加载 CSV
        df = pd.read_csv(csv_path, header=None)  # 第一列 mp-id，其他 51 列 DOS
        self.ids = df[0].tolist()  # mp-ids
        phdos_data = df.iloc[:, 1:].values.tolist()  # 51 维 DOS 列表

        element_types = get_element_list([])  # 将在加载结构后设置

        # 加载晶体结构
        self.structures = []
        for _id in tqdm(self.ids, desc="Reading CIF files"):
            cif_path = os.path.join(self.structure_dir, f"{_id}.cif")
            if os.path.exists(cif_path):
                structure = Structure.from_file(cif_path)
                self.structures.append(structure)
            else:
                print(f"Warning: CIF not found for {_id}")
                self.structures.append(None)  # 占位，稍后过滤

        # 过滤无效结构
        valid_indices = [i for i, s in enumerate(self.structures) if s is not None]
        self.structures = [self.structures[i] for i in valid_indices]
        self.ids = [self.ids[i] for i in valid_indices]
        phdos_data = [phdos_data[i] for i in valid_indices]

        # 更新 element_types
        element_types = get_element_list(self.structures)

        self.converter = Structure2Graph(element_types=element_types, cutoff=threebody_cutoff)

        # 标签 (标准化为 'phdos')
        self.labels = {"phdos": phdos_data}
        self.apply_minmax()
        
        super().__init__(
            converter=self.converter,
            threebody_cutoff=threebody_cutoff,
            structures=self.structures,
            labels=self.labels,
            save_cache=False
        )
        
    def apply_minmax(self):
        phdos_tensor = torch.tensor(self.labels["phdos"], dtype=torch.float)  # (num_samples, seq_len)
        self.phdos_min = phdos_tensor.min(dim=1, keepdim=True).values
        self.phdos_max = phdos_tensor.max(dim=1, keepdim=True).values
        phdos_norm = (phdos_tensor - self.phdos_min) / (self.phdos_max - self.phdos_min)
        self.labels["phdos"] = phdos_norm.tolist()  # 转回 list

    def reverse_minmax(self, y_pred, idx):
        """
        对预测结果反归一化
        y_pred: (seq_len,) 或 (batch_size, seq_len)
        idx: 对应样本索引，或者 batch 索引列表
        """
        if isinstance(idx, int):
            ph_min = self.phdos_min[idx]
            ph_max = self.phdos_max[idx]
        else:  # batch
            ph_min = self.phdos_min[idx]
            ph_max = self.phdos_max[idx]
        return y_pred * (ph_max - ph_min) + ph_min
    
    def __getitem__(self, idx: int):
        items = [
            self.graphs[idx],
            self.lattices[idx],
            self.state_attr[idx],
            {
                k: torch.tensor(v[idx], dtype=torch.float)
                for k, v in self.labels.items()
                if not isinstance(v[idx], str)
            },
            idx
        ]
        if self.include_line_graph:
            items.insert(2, self.line_graphs[idx])
        return tuple(items)
