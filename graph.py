import numpy as np

# 计算时间权重
def calculate_weights(ngh_ts, current_time):
    time_differences = current_time - ngh_ts
    weights = 1.0 / (1.0 + time_differences)  # 使用逆时间差作为权重
    weights[time_differences < 0] = 0  # 忽略未来的邻居
    return weights

class NeighborFinder:
    def __init__(self, adj_list, uniform=False):
        """
        Params
        ------
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """ 
       
        node_idx_l, node_ts_l, edge_idx_l, off_set_l, src_type_l, dst_type_l, edge_type_l = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l
        self.src_type_l = src_type_l
        self.dst_type_l = dst_type_l
        self.edge_type_l = edge_type_l
        
        self.off_set_l = off_set_l
        
        self.uniform = uniform
        
    def init_off_set(self, adj_list):
        """
        Params
        ------
        adj_list: List[List[int]]
        
        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        src_t_l = []
        dst_t_l = []
        edge_t_l = []
        off_set_l = [0]
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[1])
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            n_ts_l.extend([x[2] for x in curr])
            src_t_l.extend([x[3] for x in curr])
            dst_t_l.extend([x[4] for x in curr])
            edge_t_l.extend([x[5] for x in curr])           
            
            off_set_l.append(len(n_idx_l))
        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        src_t_l = np.array(src_t_l)
        dst_t_l = np.array(dst_t_l)
        edge_t_l = np.array(edge_t_l)
        off_set_l = np.array(off_set_l)

        assert(len(n_idx_l) == len(n_ts_l))
        assert(off_set_l[-1] == len(n_ts_l))
        
        return n_idx_l, n_ts_l, e_idx_l, off_set_l, src_t_l, dst_t_l, edge_t_l
        
    def find_before(self, src_idx, cut_time):
        """
    
        Params
        ------
        src_idx: int
        cut_time: float
        """
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        src_type_l = self.src_type_l 
        dst_type_l = self.dst_type_l
        edge_type_l = self.edge_type_l
        off_set_l = self.off_set_l
        
        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_src_type = src_type_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_dst_type = dst_type_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_edge_type = edge_type_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        
        if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:
            return neighbors_idx, neighbors_ts, neighbors_e_idx, neighbors_src_type, neighbors_dst_type, neighbors_edge_type

        left = 0
        right = len(neighbors_idx) - 1
        
        while left + 1 < right:
            mid = (left + right) // 2
            curr_t = neighbors_ts[mid]
            if curr_t < cut_time:
                left = mid
            else:
                right = mid
                
        if neighbors_ts[right] <= cut_time:
            return neighbors_idx[:right], neighbors_e_idx[:right], neighbors_ts[:right], neighbors_src_type[:right],  \
                    neighbors_dst_type[:right], neighbors_edge_type[:right]
        else:
            return neighbors_idx[:left], neighbors_e_idx[:left], neighbors_ts[:left], neighbors_src_type[:left],  \
                    neighbors_dst_type[:left], neighbors_edge_type[:left]

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert(len(src_idx_l) == len(cut_time_l))
        
        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_src_type_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_dst_type_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_edge_type_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        
        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_eidx, ngh_ts, ngh_src_type, ngh_dst_type, ngh_edge_type = self.find_before(src_idx, cut_time)

            if len(ngh_idx) > 0:
                if self.uniform:
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)
                    
                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]
                    out_ngh_src_type_batch[i, :] = ngh_src_type[sampled_idx]
                    out_ngh_dst_type_batch[i, :] = ngh_dst_type[sampled_idx]
                    out_ngh_edge_type_batch[i, :] = ngh_edge_type[sampled_idx]
                    
                    
                    # resort based on time
                    pos = out_ngh_t_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                    out_ngh_eidx_batch[i, :] = out_ngh_eidx_batch[i, :][pos]
                    out_ngh_src_type_batch[i, :] = out_ngh_src_type_batch[i, :][pos]
                    out_ngh_dst_type_batch[i, :] = out_ngh_dst_type_batch[i, :][pos]
                    out_ngh_edge_type_batch[i, :] = out_ngh_edge_type_batch[i, :][pos]
                elif self.weighted_sampling:
                    # 计算权重
                    weights = calculate_weights(ngh_ts, cut_time)
                    weights /= weights.sum()  # 标准化权重

                    # 加权采样
                    sampled_indices = np.random.choice(len(ngh_idx), size=num_neighbors, p=weights)
                    
                    out_ngh_node_batch[i, :] = ngh_idx[sampled_indices]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_indices]
                    out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_indices]
                    out_ngh_src_type_batch[i, :] = ngh_src_type[sampled_indices]
                    out_ngh_dst_type_batch[i, :] = ngh_dst_type[sampled_indices]
                    out_ngh_edge_type_batch[i, :] = ngh_edge_type[sampled_indices]


                else:
                    ngh_ts = ngh_ts[:num_neighbors]
                    ngh_idx = ngh_idx[:num_neighbors]
                    ngh_eidx = ngh_eidx[:num_neighbors]
                    ngh_src_type = ngh_src_type[:num_neighbors]
                    ngh_dst_type = ngh_dst_type[:num_neighbors]
                    ngh_edge_type = ngh_edge_type[:num_neighbors]
                    
                    assert(len(ngh_idx) <= num_neighbors)
                    assert(len(ngh_ts) <= num_neighbors)
                    assert(len(ngh_eidx) <= num_neighbors)
                    assert(len(ngh_src_type) <= num_neighbors)
                    assert(len(ngh_dst_type) <= num_neighbors)
                    assert(len(ngh_edge_type) <= num_neighbors)
                    
                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eidx_batch[i,  num_neighbors - len(ngh_eidx):] = ngh_eidx
                    out_ngh_src_type_batch[i, num_neighbors - len(ngh_src_type):] = ngh_src_type
                    out_ngh_dst_type_batch[i, num_neighbors - len(ngh_dst_type):] = ngh_dst_type
                    out_ngh_edge_type_batch[i, num_neighbors - len(ngh_edge_type):] = ngh_edge_type
                    
        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch, out_ngh_src_type_batch, \
            out_ngh_dst_type_batch, out_ngh_edge_type_batch

    def find_k_hop(self, k, src_idx_l, cut_time_l, num_neighbors=20):
        """Sampling the k-hop sub graph
        """
        x, y, z, i, o, u = self.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors)
        node_records = [x]
        eidx_records = [y]
        t_records = [z]
        src_type_records = [i]
        dst_type_records = [o]
        edge_type_records = [u]
        for _ in range(k -1):
            ngn_node_est, ngh_t_est = node_records[-1], t_records[-1] # [N, *([num_neighbors] * (k - 1))]
            orig_shape = ngn_node_est.shape
            ngn_node_est = ngn_node_est.flatten()
            ngn_t_est = ngh_t_est.flatten()
            out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch, out_ngh_src_type_batch, \
            out_ngh_dst_type_batch, out_ngh_edge_type_batch = self.get_temporal_neighbor(ngn_node_est, ngn_t_est, num_neighbors)
            out_ngh_node_batch = out_ngh_node_batch.reshape(*orig_shape, num_neighbors) # [N, *([num_neighbors] * k)]
            out_ngh_eidx_batch = out_ngh_eidx_batch.reshape(*orig_shape, num_neighbors)
            out_ngh_t_batch = out_ngh_t_batch.reshape(*orig_shape, num_neighbors)
            out_ngh_src_type_batch = out_ngh_src_type_batch.reshape(*orig_shape, num_neighbors)
            out_ngh_dst_type_batch = out_ngh_dst_type_batch.reshape(*orig_shape, num_neighbors)
            out_ngh_edge_type_batch = out_ngh_edge_type_batch.reshape(*orig_shape, num_neighbors)

            node_records.append(out_ngh_node_batch)
            eidx_records.append(out_ngh_eidx_batch)
            t_records.append(out_ngh_t_batch)
            src_type_records.append(out_ngh_src_type_batch)
            dst_type_records.append(out_ngh_dst_type_batch)
            edge_type_records.append(out_ngh_edge_type_batch)
        return node_records, eidx_records, t_records, src_type_records, dst_type_records, edge_type_records

            

