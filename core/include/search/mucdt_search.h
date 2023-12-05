/**
 * @file mucdt_search.h
 * @brief The MUCDT search algorithm.
 */

#ifndef MUCDT_SEARCH_H
#define MUCDT_SEARCH_H

#include <forward_list>
#include <vector>
#include <array>

#include "cache/approx_bitset_cache.h"
#include "data/data_manager.h"
#include "solution/solution.h"
#include "solution/decision_tree.h"
#include "search/search_configuration.h"

struct AndNode;

struct OrNode {
  double trueValue;
  double stopValue;
  double totalValueEstimate;
  size_t numVisits;
  std::vector<size_t> numVisitsToEachChild;
  std::vector<size_t> numVisitsToEachActionInSubtree;
  std::vector<double> childrenTotalValueEstimates;
  std::vector<double> actionsTotalValueEstimates;
  double beta() const;
  double actionSubtreeValueEstimate(size_t action) const;
  double UCB1ValueEstimate(size_t action) const;
};

struct AndNode {
  size_t feature;
  OrNode *left;
  OrNode *right;
  OrNode *parent;
};

class MUCDTSearch {
  public:
    MUCDTSearch(const DataManager& dm, const SearchConfiguration& cfg) :
      dm_(dm),
      cfg_(cfg),
      subproblem_(dm_),
      cache_(NUM_BLOCKS(dm.getNumSamples())),
      rootNode_(buildNode(subproblem_.getLabelCounts(), 0))
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

    OrNode *buildNode(const std::array<int, 2>& labelCounts, size_t depth);
    OrNode *findExpansion();
    void expand(OrNode *node);
    void backpropagate(OrNode *source);
    DecisionTree *buildDecisionTree(OrNode *node);
};

#endif