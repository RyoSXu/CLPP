# utils/collate.py
import torch
import dgl

def collate_fn_contrastive(batch):
    graphs, lattices, state_attr, labels, idxs = map(list, zip(*batch))
    
    for i, g in enumerate(graphs):
        frac_coords = g.ndata['frac_coords']
        lattice_i = lattices[i].squeeze(0)
        pos = frac_coords @ lattice_i.T
        g.ndata['pos'] = pos
        g.edata['pbc_offshift'] = g.edata['pbc_offset']
    
    g = dgl.batch(graphs)
    phdos_list = torch.stack([lbl['phdos'] for lbl in labels])
    idxs = torch.tensor(idxs, dtype=torch.long)
    return g, phdos_list, idxs
