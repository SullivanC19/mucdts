/**
 * @file search_bindings.cpp
 * @brief Set up bindings for search functions to Python with pybind11.
 *
 * This file contains the bindings for the search functions to Python with
 * pybind11. The bindings are exported as a Python module called mucdts.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <array>
#include <chrono>

#include "data/data_manager.h"
#include "solution/solution.h"
#include "search/search_configuration.h"
#include "search/mucdt_search.h"

namespace py = pybind11;

Solution mucdtSearch(
    std::vector<std::vector<bool>> features,
    std::vector<bool> labels,
    double exploration,
    int numExpansions,
    double sparsity,
    double k
  )
{
    DataManager dm(features, labels);
    SearchConfiguration cfg {exploration, static_cast<size_t>(numExpansions), sparsity, k};
    MUCDTSearch srch(dm, cfg);
    Solution res = srch.search();
    return res;
};

PYBIND11_MODULE(mucdts, m) {
    m.doc() = "MAP tree search binding";

    m.def(
        "search",
        &mucdtSearch,
        "MUCDT Search",
        py::arg("features"),
        py::arg("labels"),
        py::arg("exploration"),
        py::arg("numExpansions"),
        py::arg("sparsity"),
        py::arg("k")
    );

    py::class_<Solution>(m, "Solution").def_readwrite("tree", &Solution::treeRepresentation);
};