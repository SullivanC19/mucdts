#include "search/mucdt_search.h"

#include <queue>
#include <set>

float beta(size_t N, float k) {
  return std::sqrt(k / (3 * N + k));
};

std::vector<float> UCB1ValueEstimate(
    size_t N,
    const std::vector<size_t>& Na,
    const std::vector<float>& Va,
    const std::vector<float>& Ra,
    float beta,
    float c) {
  std::vector<float> UCB1s = std::vector<float>(Na.size());

  for (size_t i = 0; i < Na.size(); i++) {
    if (Na[i] == 0) {
      UCB1s[i] = 1000.0;
    } else {
      UCB1s[i] = (beta * Ra[i]) + (1 - beta) * Va[i] + c * std::sqrt(std::log(N) / Na[i]);
    }
  }
  return UCB1s;
};

Solution MUCDTSearch::search() {
  for (size_t i = 0; i < cfg_.numExpansions; i++) {
    findTree(rootNode_);
  }
  DecisionTree *dt = buildDecisionTree(rootNode_);
  Solution sol = Solution{dt->toString()};
  delete dt;
  return sol;
};

OrNode *MUCDTSearch::buildNode(const std::array<int, 2>& labelCounts) {
  OrNode *node = new OrNode();

  node->stopValue = std::max(
    static_cast<float>(labelCounts[0]) / (2 * dm_.getNumNegativeLabels()),
    static_cast<float>(labelCounts[1]) / (2 * dm_.getNumPositiveLabels()));

  node->trueValue = node->stopValue;
  node->totalValueEstimate = 0.0;
  node->numVisits = 0;
  node->parents = std::forward_list<AndNode *>();
  node->numVisitsToEachActionInSubtree = std::vector<size_t>(dm_.getNumFeatures(), 0);
  node->actionsTotalValueEstimates = std::vector<float>(dm_.getNumFeatures(), 0.0);
  node->children = std::vector<AndNode *>(dm_.getNumFeatures(), nullptr);
  orNodes_.push_front(node);
  return node;
};

std::vector<float> MUCDTSearch::getValidSplitUCB1Values(OrNode *node, const std::vector<size_t>& validSplits) const {
  size_t N = node->numVisits;
  std::vector<size_t> Na(validSplits.size(), 0.0);
  std::vector<float> Va(validSplits.size(), 0.0);
  std::vector<float> Ra(validSplits.size(), 0.0);
  size_t i = 0;
  for (size_t f : validSplits) {
    Ra[i] = node->actionsTotalValueEstimates[f];
    if (node->numVisitsToEachActionInSubtree[f] > 0) {
      Ra[i] /= node->numVisitsToEachActionInSubtree[f];
    }
    AndNode *split = node->children[f];
    if (split != nullptr && split->left != nullptr) {
      Na[i] += split->left->numVisits;
      if (split->left->numVisits > 0) {
        Va[i] += split->left->totalValueEstimate / split->left->numVisits;
      }
    }
    if (split != nullptr && split->right != nullptr) {
      Na[i] += split->right->numVisits;
      if (split->right->numVisits > 0) {
        Va[i] += split->right->totalValueEstimate / split->right->numVisits;
      }
    }
    Va[i] -= cfg_.sparsity;
    i++;
  }

  std::vector<float> ucb1s = UCB1ValueEstimate(N, Na, Va, Ra, beta(node->numVisits, cfg_.k), cfg_.exploration);
  return ucb1s;
}

void propagateImprovedTree(OrNode *node, float sparsity) {
  std::queue<std::pair<OrNode *, int>> q;
  std::set<OrNode *> visited;
  q.push({node, -1});
  visited.insert(node);
  
  while (!q.empty()) {
    std::pair<OrNode *, int> top = q.front();
    q.pop();
    OrNode *node = top.first;
    int action = top.second;

    bool queueParents = action == -1;
    if (action >= 0) {
      AndNode* updatedSplit = node->children[action];
      if (updatedSplit->left == nullptr || updatedSplit->right == nullptr) {
        continue;
      }

      double updatedValue = updatedSplit->left->trueValue + updatedSplit->right->trueValue - sparsity;
      if (updatedValue > node->trueValue) {
        node->trueValue = updatedValue;
        queueParents = true;
      }
    }

    if (queueParents) {
      for (AndNode *parent : node->parents) {
        if (visited.find(parent->parent) == visited.end()) {
          q.push({parent->parent, parent->feature});
          visited.insert(parent->parent);
        }
      }
    }
  }
}

std::pair<float, std::vector<std::pair<size_t, float>>> MUCDTSearch::findTree(OrNode *node) {
  node->numVisits++;

  std::vector<size_t> validSplits = subproblem_.getValidSplits();

  // stop if no valid splits or first visit
  if (node->numVisits == 1 || validSplits.empty()) {
    node->totalValueEstimate += node->stopValue;
    return {node->totalValueEstimate / node->numVisits, {}};
  }

  std::vector<float> ucb1s = getValidSplitUCB1Values(node, validSplits);

  size_t bestSplitInd = std::distance(ucb1s.begin(), std::max_element(ucb1s.begin(), ucb1s.end()));
  float bestSplitUCB1 = ucb1s[bestSplitInd];
  size_t bestSplit = validSplits[bestSplitInd];
  
  // stop if better than splitting
  if (bestSplitUCB1 < node->stopValue) {
    node->totalValueEstimate += node->stopValue;
    return {node->totalValueEstimate / node->numVisits, {}};
  }

  // if split is null, create new one
  AndNode *split = node->children[bestSplit];
  if (split == nullptr) {
    split = (node->children[bestSplit] = new AndNode());
    split->parent = node;
    split->feature = bestSplit;
  }

  std::pair<float, std::vector<std::pair<size_t, float>>> leftTree;
  std::pair<float, std::vector<std::pair<size_t, float>>> rightTree;

  subproblem_.applySplit(bestSplit, false);
  if (split->left == nullptr) {
    OrNode *cached = static_cast<OrNode *>(cache_.get(subproblem_));
    if (cached == nullptr) {
      cached = buildNode(subproblem_.getLabelCounts());
      cache_.put(subproblem_, cached);
    }
    cached->parents.push_front(node->children[bestSplit]);
    split->left = cached;
  }
  leftTree = findTree(split->left);
  subproblem_.revertSplit();

  subproblem_.applySplit(bestSplit, true);
  if (split->right == nullptr) {
    OrNode *cached = static_cast<OrNode *>(cache_.get(subproblem_));
    if (cached == nullptr) {
      cached = buildNode(subproblem_.getLabelCounts());
      cache_.put(subproblem_, cached);
    }
    cached->parents.push_front(node->children[bestSplit]);
    split->right = cached;
  }
  rightTree = findTree(split->right);
  subproblem_.revertSplit();

  float value = leftTree.first + rightTree.first - cfg_.sparsity;
  node->totalValueEstimate += value;

  float trueValue = split->left->trueValue + split->right->trueValue - cfg_.sparsity;
  if (trueValue > node->trueValue) {
    node->trueValue = trueValue;
    propagateImprovedTree(node, cfg_.sparsity);
  }

  std::vector<std::pair<size_t, float>> actionValues(leftTree.second.size() + rightTree.second.size() + 1);
  actionValues[0] = {bestSplit, value};
  std::copy(leftTree.second.begin(), leftTree.second.end(), actionValues.begin() + 1);
  std::copy(rightTree.second.begin(), rightTree.second.end(), actionValues.begin() + 1 + leftTree.second.size());

  for (std::pair<size_t, float> entry : actionValues) {
    size_t action = entry.first;
    float value = entry.second;
    node->actionsTotalValueEstimates[action] += value;
    node->numVisitsToEachActionInSubtree[action]++;
  }

  return {node->totalValueEstimate / node->numVisits, actionValues};
}

DecisionTree *MUCDTSearch::buildDecisionTree(OrNode *node) {
  std::array<int, 2> labelCounts = subproblem_.getLabelCounts();
  assert(node->stopValue == std::max(labelCounts[0], labelCounts[1]) / dm_.getNumSamples());

  if (node == rootNode_) {
    std::cout << "root: " << node->trueValue << " " << node->stopValue << " " << node->totalValueEstimate << " " << node->numVisits << std::endl;
  }

  // no possible splits — return leaf
  if (node->trueValue == node->stopValue) {
    return new DecisionTree();
  }

  float bestValue = node->stopValue;
  size_t bestFeature = 0;
  for (size_t f = 0; f < node->children.size(); f++) {
    if (node->children[f] == nullptr) continue;
    float trueChildValue = node->children[f]->left->trueValue + node->children[f]->right->trueValue - cfg_.sparsity;
    if (trueChildValue > bestValue) {
      bestValue = trueChildValue;
      bestFeature = f;
    }
  }

  if (bestValue == node->stopValue) {
    return new DecisionTree();
  }

  if (std::abs(bestValue - node->trueValue) > 0.0001) {
    std::cout << "Warning: best value " << bestValue << " not equal to node true value " << node->trueValue << std::endl;
  }

  subproblem_.applySplit(bestFeature, false);
  DecisionTree *left = buildDecisionTree(node->children[bestFeature]->left);
  subproblem_.revertSplit();

  subproblem_.applySplit(bestFeature, true);
  DecisionTree *right = buildDecisionTree(node->children[bestFeature]->right);
  subproblem_.revertSplit();

  return new DecisionTree(bestFeature, left, right);

  return nullptr;
};
