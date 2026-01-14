# VPTQ 量化算法代码定位与流程

- **分层入口与 Hessian 载入**：`vptq/layer_quantizer.py` 第18-190行。对每个 Transformer 层遍历权重线性层，调用 `load_hessian`/`load_inv_hessian` 读取预计算 Hessian 与逆矩阵（第31-62行），据此构造 `NPVectorQuantizer`（第69-85行）与 `VPTQ` 算子（第88-103行），并执行 `fast_vector_quant()` 完成该层的量化（第105行）。量化完成后使用 `VQuantLinear` 封装量化参数并替换原始线性层（第135-183行）。
- **权重预处理与初始化聚类**：`vptq/vptq.py` 的 `fast_vector_quant`（第120-319行）完成核心流程：按需对权重做归一化和转置、根据 Hessian 零对角处理及置零（第124-176行），依据 `NPVectorQuantizer.get_group_setting` 计算分组与 outlier 列（第151-153行），用 Hessian 对角线作为 KMeans 权重（第160-170行），可选按对角排序执行列置换（第179-191行），随后触发 `init_centroids_indices` 用 cuML KMeans 初始化码本并得到初始量化权重（第205-221行）。
- **首轮 VPTQ 误差补偿**：`vptq/vptq.py` 中的 `vptq` 函数（第321-427行）按 block 遍历列（第351-420行），调用 `quantize_vector` 将每个子向量映射到码本（第367-372行），再结合逆 Hessian 对剩余未量化列执行误差回传与权重更新（第377-420行），得到首轮量化权重与误差张量。
- **残差码本与二次量化**：若配置了残差码本，`fast_vector_quant` 会先在首轮误差上初始化残差码本 `init_res_centroids_indices`（第254-267行），随后清空索引并再次调用 `vptq(..., enable_residual=True)` 进行残差量化与误差补偿（第269-293行），再将置换与归一化反映射回原始形状（第297-318行）。
- **向量量化细节实现**：`vptq/quantizer.py` 定义了 `NPVectorQuantizer`。初始化阶段设置 outlier/主码本长度、分组与置换/归一化开关（第75-158行）；`get_group_setting` 和 `get_index_list` 依据 n% outlier 与 group_num/group_size 生成分段（第209-253行）；`init_centroids_indices` 使用 cuML KMeans 在各分段上训练码本并记录索引（第255-320行）；`quantize_vector` 按列切片、计算到码本的 cdist 并保存索引（第322-374行）；`init_res_centroids_indices` 与 `quantize_residual_vector` 对误差执行残差码本训练与二次量化（第376-496行）。
- **Hessian 预处理**：`vptq/utils/hessian.py`（第10-58行）将磁盘中的压缩 Hessian 展开为对称矩阵并执行正则化处理（`flat_to_sym`、`regularize_H`、`basic_preprocess`），为后续加权 KMeans 与误差度量提供输入。
- **多 GPU 调度与日志**：`vptq/quantize_executer.py`（第45-102行）将待量化层分配到不同 GPU、在每个进程中调用 `layer_quantizer`，并将量化后参数或元信息写回队列或磁盘，属于算法执行层的调度代码。
