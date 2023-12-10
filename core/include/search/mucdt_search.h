/**
 * @file mucdt_search.h
 * @brief The MUCDT search algorithm.
 */

#ifndef MUCDT_SEARCH_H
#define MUCDT_SEARCH_H

#include <forward_list>
#include <vector>
#include <utility>
#include <array>

#include "cache/approx_bitset_cache.h"
#include "data/data_manager.h"
#include "solution/solution.h"
#include "solution/decision_tree.h"
#include "search/search_configuration.h"

struct AndNode;

struct OrNode {
  float trueValue;
  float stopValue;
  float totalValueEstimate;
  size_t numVisits;
  std::vector<size_t> numVisitsToEachActionInSubtree;
  std::vector<float> actionsTotalValueEstimates;
  std::vector<AndNode *> children;
  std::forward_list<AndNode *> parents;
};

struct AndNode {
  OrNode *parent;
  size_t feature;
  OrNode *left;
  OrNode *right;
};

class MUCDTSearch {
  public:
    MUCDTSearch(const DataManager& dm, const SearchConfiguration& cfg) :
      dm_(dm),
      cfg_(cfg),
      subproblem_(dm_),
      cache_(NUM_BLOCKS(dm.getNumSamples())),
      rootNode_(buildNode(subproblem_.getLabelCounts()))
      {}
    Solution search();
  private:
    const DataManager& dm_;
    const SearchConfiguration& cfg_;
    Subproblem subproblem_;
    ApproxBitsetCache cache_;
    OrNode *rootNode_;
    std::forward_list<OrNode *> orNodes_ = std::forward_list<OrNode *>();
    std::forward_list<AndNode *> andNodes_ = std::forward_list<AndNode *>();

    OrNode *buildNode(const std::array<int, 2>& labelCounts);
    std::pair<float, std::vector<std::pair<size_t, float>>> findTree(OrNode *node);
    std::vector<float> getValidSplitUCB1Values(OrNode *node, const std::vector<size_t>& validSplits) const;
    DecisionTree *buildDecisionTree(OrNode *node);
};

#endif