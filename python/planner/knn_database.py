import time

import faiss
import torch
import faiss.contrib.torch_utils


class KNNDatabase:
    def __init__(self, points: torch.Tensor, exact: bool = True, ivf_nlist: int = 100):
        assert len(points.shape) == 2
        ivf_nlist = max(1, min(ivf_nlist, points.shape[0] // 39))
        quantizer = faiss.IndexFlatL2(points.shape[1])
        if not exact:
            self.__index = faiss.IndexIVFFlat(quantizer, points.shape[1], ivf_nlist)
        else:
            self.__index = quantizer
        if points.is_cuda:
            res = faiss.StandardGpuResources()
            self.__index = faiss.index_cpu_to_gpu(res, 0, self.__index)
            self.__index.train(points)
            self.__index.add(points)
        else:
            points_np = points.detach().numpy()
            self.__index.train(points_np)
            self.__index.add(points_np)

    def search(self, query_points: torch.Tensor, k: int = 10):
        if isinstance(self.__index, faiss.IndexIVFFlat):
            _, idx_np = self.__index.search(query_points.detach().numpy(), k)
            idx = torch.from_numpy(idx_np)
        else:
            _, idx = self.__index.search(query_points, k)
        return idx
