// graph_pybind.h — S4 expose minimal graph API to Python
//
// 暴露给 Python 的最小集 (跟 feature-auto-batch 的 rpu_backend.cpp 一致):
//   - GraphSignature: 9 个 readwrite field 给 Python 构造 / 比较 / hash
//   - Graph (= RpuKernelGraph): begin(sig) / end / abort / state / graph_size
//   - GraphCache (= RpuGraphCache): default ctor / get_or_create / size / clear
//
// 高级特性(child graph / fast replay / register patch / signature tree /
// host callback / stats)不在 S4 范围内,等 S5 港时按需追加。
//
// Hook 入口:src/core/rpu_backend.cpp 的 PYBIND11_MODULE 内 call
// graph_pybind::add_bindings(m)。

#pragma once

#include <pybind11/pybind11.h>

namespace graph_pybind {
void add_bindings(pybind11::module_& m);
}
