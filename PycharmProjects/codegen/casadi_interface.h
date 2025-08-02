#pragma once
#include <vector>
#include <memory>
#include <cassert>

using int_T = long long int;

template <typename T>
class CasadiFunction {
public:
    CasadiFunction(
        int (*f)(const T**, T**, int_T*, T*, int),
        const int_T* (*sp_out)(int_T),
        int (*work)(int_T*, int_T*, int_T*, int_T*)
    );
    
    std::vector<std::vector<T>> evaluate(const std::vector<std::vector<T>>& inputs);
    
private:
    int (*f_)(const T**, T**, int_T*, T*, int) = nullptr;
    const int_T* (*sp_out_)(int_T) = nullptr;
    
    int_T sz_arg_ = 0;
    int_T sz_res_ = 0;
    int_T sz_iw_ = 0;
    int_T sz_w_ = 0;
    
    struct OutputSparsity {
        int nrows;
        int ncols;
        std::vector<int_T> col_ptr;
        std::vector<int_T> row_idx;
    };
    std::vector<OutputSparsity> sparsity_info_;
    
    std::unique_ptr<int_T[]> iw_;
    std::unique_ptr<T[]> w_;
};