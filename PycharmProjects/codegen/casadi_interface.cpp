#include "casadi_interface.h"
#include <algorithm>

template <typename T>
CasadiFunction<T>::CasadiFunction(
    int (*f)(const T**, T**, int_T*, T*, int),
    const int_T* (*sp_out)(int_T),
    int (*work)(int_T*, int_T*, int_T*, int_T*)
) : f_(f), sp_out_(sp_out) 
{
    // 1. 初始化工作空间
    work(&sz_arg_, &sz_res_, &sz_iw_, &sz_w_);
    iw_ = std::make_unique<int_T[]>(sz_iw_);
    w_ = std::make_unique<T[]>(sz_w_);

    // 2. 解析稀疏模式
    sparsity_info_.resize(sz_res_);
    for (int_T i = 0; i < sz_res_; ++i) {
        const int_T* sp = sp_out_(i);
        OutputSparsity info;
        info.nrows = static_cast<int>(sp[0]);
        info.ncols = static_cast<int>(sp[1]);
        
        // 提取列指针和行索引
        const int_T* col_ptr_start = sp + 2;
        info.col_ptr.assign(col_ptr_start, col_ptr_start + info.ncols + 1);
        
        const int_T nnz = info.col_ptr[info.ncols];
        const int_T* row_idx_start = col_ptr_start + (info.ncols + 1);
        info.row_idx.assign(row_idx_start, row_idx_start + nnz);
        
        sparsity_info_[i] = std::move(info);
    }
}

template <typename T>
std::vector<std::vector<T>> CasadiFunction<T>::evaluate(const std::vector<std::vector<T>>& inputs) 
{
    // 1. 验证输入
    if (inputs.size() != static_cast<size_t>(sz_arg_)) {
        throw std::invalid_argument("输入数量不匹配");
    }
    
    // 2. 准备指针数组
    std::vector<const T*> arg_ptrs;
    std::vector<T*> res_ptrs;
    arg_ptrs.reserve(sz_arg_);
    res_ptrs.reserve(sz_res_);
    
    // 3. 处理输入（使用常量引用）
    for (const auto& input : inputs) {
        arg_ptrs.push_back(input.data());
    }
    
    // 4. 准备输出缓冲区
    std::vector<std::vector<T>> outputs;
    outputs.reserve(sz_res_);
    for (const auto& info : sparsity_info_) {
        const int output_size = info.nrows * info.ncols;
        outputs.emplace_back(output_size, T(0)); // 初始化为0
        res_ptrs.push_back(outputs.back().data());
    }
    
    // 5. 执行计算
    f_(arg_ptrs.data(), res_ptrs.data(), iw_.get(), w_.get(), 1);
    
    // 6. 处理稀疏输出
    for (int i = 0; i < sz_res_; ++i) {
        const auto& info = sparsity_info_[i];
        const int_T nnz = info.col_ptr[info.ncols];
        
        // 直接操作vector（值传递）
        for (int_T k = 0; k < nnz; ++k) {
            const int_T col = std::distance(
                info.col_ptr.begin(),
                std::upper_bound(info.col_ptr.begin(), info.col_ptr.end(), k)
            ) - 1;
            const int_T row = info.row_idx[k];
            outputs[i][row + info.nrows * col] = res_ptrs[i][k];
        }
    }
    
    return outputs;
}

// 单一定义显式实例化
template class CasadiFunction<double>;