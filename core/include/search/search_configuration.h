/**
 * @file search_configuration.h
 * @brief Stores the configuration for MUCDT search.
 */

#ifndef SEARCH_CONFIGURATION_H
#define SEARCH_CONFIGURATION_H

struct SearchConfiguration {
    double exploration;
    size_t numExpansions;
    double sparsity;
    double k;
};

#endif