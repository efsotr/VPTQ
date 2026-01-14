# VPTQ 算法的 outlier 选取方法

本文简要总结当前代码中 outlier 的划分与使用逻辑，方便快速查阅。

## 1. outlier 数量的确定
- 参数 `npercent` 表示有多少列被当作 outlier。  
- 在 `vptq/quantizer.py::NPVectorQuantizer.get_group_setting` 中，先计算  
  `initial_outlier_size = math.ceil(npercent / 100 * data.shape[1])`。  
- 若启用了分组（`group_num` >= 1），会将剩余列数补齐到能被 `group_num` 整除：  
  `remaining = data.shape[1] - initial_outlier_size`，如果 `remaining % group_num != 0`，则把余数部分并入 outlier，使 outlier 区间略大于 `npercent`。  
- 最终得到的 `outlier_size`、`group_num` 和推导出的 `group_size` 会在后续量化中复用。

## 2. outlier 列的选择顺序
- 在 `vptq/vptq.py::VPTQ.fast_vector_quant` 中，当 `quantizer.enable_perm` 为真时，会按 Hessian 对角线从大到小排序生成 `perm`，并据此对权重和 Hessian 重新排序（`algorithm.md` 的已知问题中提示该开关尚未充分验证，使用时需注意）。  
- 之后的分区操作都基于排序后的列，因此 **outlier 区间对应 Hessian 对角线值最大的列（敏感度最高的列）**；若未开启 `enable_perm`，则直接按照原始列顺序取前 `outlier_size` 列作为 outlier。

## 3. 划分区间与量化方式
- `vptq/quantizer.py::NPVectorQuantizer.get_index_list` 会构造 `[0, outlier_size)` 作为 outlier 区间，剩余部分按 `group_size` 均分，形成 `1 + group_num` 个子区间。  
- outlier 区间使用单独的向量长度和码本：  
  - 向量长度取 `vector_lens[0]`（≤0 则关闭 outlier 功能）；  
  - 码本大小取 `num_centroids[0]`；  
  - 当前实现不支持 outlier 残差码本（`num_res_centroids[0]` 需为 -1，见 `vptq/layers/vqlinear.py` 第 119 行的断言）。  
- KMeans 初始化时，如果选择 `kmeans_mode='hessian'`，样本权重会使用 Hessian 对角线（在 `vptq/quantizer.py::NPVectorQuantizer.init_centroids_indices` 中传入 `kmeans_weight`），让聚类更关注敏感列。

综上，outlier 的选取流程为：按照 Hessian 重要性（或原始顺序）排列列向量 → 取 `npercent`（加上补齐余数）对应的前若干列作为 outlier → 为该区间使用独立的向量长度与码本进行量化，其余列走正常分组量化流程。
