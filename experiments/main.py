import numpy as np
from searchers.mucdts import run as mucdts_run
from searchers.dl85 import run as dl85_run
# from searchers.gosdt import run as gosdt_run
import matplotlib.pyplot as plt

MORTALITY_DATA_PATH = 'data/mortality.csv'
NUM_FEATURES = 51
SKIP_FEATURES = 2
NUM_BUCKETS_PER_FEATURE = 8
REGULARIZATION = 0.001
TL = [1, 5, 10, 20, 30, 40, 60, 80, 100, 130, 160, 200]
DEPTHS = [1, 2, 3, 4, 5]
T = [3e2, 1e3,3e3, 1e4, 3e4, 6e4, 1e5, 2e5, 3e5, 4e5, 5e5]
C = 2
K = 1000


def load_and_binarize_mortality_data():
  data = np.loadtxt(MORTALITY_DATA_PATH, delimiter=',', skiprows=1, usecols=range(SKIP_FEATURES, NUM_FEATURES), dtype=str)
  data[data == 'NA'] = '0'
  data = data.astype(float)
  threshold_inds = np.round(np.linspace(0, data.shape[0], NUM_BUCKETS_PER_FEATURE + 1)).astype(int)[:-1]
  bdata = np.zeros((data.shape[0], 0), dtype=bool)
  for col in range(data.shape[1]):
    thresholds = np.sort(np.unique(data[:, col]))[:-1]
    if len(thresholds) > NUM_BUCKETS_PER_FEATURE:
      thresholds = np.unique(np.sort(data[:, col])[threshold_inds])
    bdata_col = data[:, col][:, None] > thresholds[None, :]
    bdata = np.hstack((bdata, bdata_col))
  return bdata

if __name__ == '__main__':
  data = load_and_binarize_mortality_data()
  X = data[:, 1:].astype(int)
  y = data[:, 0].astype(int)

  mucdts_scores = []
  mucdts_times = []

  for t in T:
    print(f't = {t}')
    tree, time = mucdts_run(X, y, C, K, REGULARIZATION, int(t))
    print(f'time = {time}')
    print(tree)
    print(f'bacc = {tree.balanced_accuracy(X, y)}')
    print(f'size = {tree.size()}')

    score = tree.balanced_accuracy(X, y) - (tree.size() - 1) / 2 * REGULARIZATION
    print(f'score = {score}')
    mucdts_scores.append(score)
    mucdts_times.append(time)
    print()

  dl85_scores = dict()
  dl85_times = dict()
  for depth in DEPTHS:
    dl85_scores[depth] = []
    dl85_times[depth] = []
    for tl in TL:
      print(f'tl = {tl}, depth = {depth}')
      tree, time, timeout = dl85_run(X, y, regularization=REGULARIZATION, time_limit=tl, depth=depth)
      print(f'time = {time}')
      print(tree)
      print(f'bacc = {tree.balanced_accuracy(X, y)}')
      print(f'size = {tree.size()}')
      score = tree.balanced_accuracy(X, y) - (tree.size() - 1) / 2 * REGULARIZATION
      print(f'score = {score}')
      dl85_scores[depth].append(score)
      dl85_times[depth].append(time)

  # GOSDT bugs: (1) miscalculation of balanced accuracy, (2) infinite recursion for decoding large trees

  # for tl in TL:
  #   print(f'tl = {tl}')
  #   tree, time, timeout = gosdt_run(X, y, regularization=REGULARIZATION, time_limit=tl)
  #   print(f'time = {time}')
  #   print(tree)
  #   print(f'bacc = {tree.balanced_accuracy(X, y)}')
  #   print(f'size = {tree.size()}')

  #   score = tree.balanced_accuracy(X, y) - (tree.size() - 1) / 2 * REGULARIZATION
  #   print(f'score = {score}')
  #   gosdt_scores.append(score)
  #   print()

  plt.plot(mucdts_times, mucdts_scores, label='MUCDTS')
  for depth in DEPTHS:
    plt.plot(dl85_times[depth], dl85_scores[depth], label=f'DL (depth={depth})')
  plt.xlabel('Time')
  plt.ylabel('Score')
  plt.xscale('log')
  plt.title('MUCDTS vs DL8.5 on Mortality Dataset')
  plt.legend()
  plt.savefig('mortality.pdf', bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
