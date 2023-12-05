#include "search/mucdt_search.h"

// rapid action value estimation bias decays over time, SHOULD NOT INCLUDE PREVIOUSLY SPLIT FEATURES
// rollout is just stopping

// use MCTS+RAVE to find next expansion (choose split based on mcts-rave, side based on squared ratio of num training points (b.c. of variance))
// backpropagate best solution
// backpropagate value of expansion
// update action value estimate
// repeat

// on expansion
// - IF NODE EXISTS, ADD AN EDGE AND KEEP GOING!!!
// - add both left subtree and right subtree to AND/OR search graph
// - set true value of both to stop value
// - set number of visits to 1 and total value estimate to stop value
// - set number of visits to each action in subtree to 0 and total value estimate to 0


// node should store:
// - true maximum value
// - stop value
// - total value estimate
// - number of visits
// - number of visits to each child
// - number of visits to each action in subtree
// - actions total value estimates
// - children total value estimates

// features (X), labels (y), exploration (c), time (T), sparsity (s), beta parameter (k), beta decay (gamma)
// beta(s) = sqrt(k/(3N(s) + k))

double OrNode::beta() const {
  return 0;
};

double OrNode::actionSubtreeValueEstimate(size_t action) const {
  return 0;
};

double OrNode::UCB1ValueEstimate(size_t action) const {
  return 0;
};

Solution MUCDTSearch::search() {
  return Solution();
};

OrNode *MUCDTSearch::buildNode(const std::array<int, 2>& labelCounts, size_t depth) {
  return nullptr;
};

OrNode *MUCDTSearch::findExpansion() {
  return nullptr;
};

void MUCDTSearch::expand(OrNode *node) {

};

void MUCDTSearch::backpropagate(OrNode *source) {

};

DecisionTree *MUCDTSearch::buildDecisionTree(OrNode *node) {
  return nullptr;
};
