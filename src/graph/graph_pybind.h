// Python bindings for the public GraphSignature, Graph, and GraphCache APIs.
// The extension module installs them by calling graph_pybind::add_bindings().

#pragma once

#include <pybind11/pybind11.h>

namespace graph_pybind {
void add_bindings(pybind11::module_& m);
}
