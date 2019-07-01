import minishogilib

import math
import copy
import collections
import numpy as np
from operator import itemgetter

class Node:
  def __init__(self, policy=0):
    self.N = 0
    self.V = 0
    self.P = policy
    self.W = 0
    self.children = {}

  def get_puct(self, parent_N):
    c_base = 19652
    c_init = 1.25

    C = ((1 + self.N + c_base) / c_base) + c_init

    Q = 0 if self.N == 0 else self.W / self.N
    U = C * self.P * math.sqrt(parent_N) / (1 + self.N)

    return (Q + U)

  def expanded(self):
    return len(self.children) > 0

def select_child(node):
  _, move, child = max(((child.get_puct(node.N), move, child) for move, child in node.children.items()), key=itemgetter(0))

  return move, child

def evaluate(node, position):
  moves = position.generate_moves(True, True)

  # ToDo: Use neural network
  value, policy = np.random.uniform(0, 1), np.random.uniform(0, 1, len(moves))

  # sef value and policy
  node.V = value
  for i, move in enumerate(moves):
    node.children[move] = Node(policy[i])

  return value

def backpropagate(position, search_path, value):
  flip = False
  while True:
    node = search_path.pop()

    node.W += value if not flip else (1 - value)
    node.N += 1

    if len(search_path) == 0:
      break

    position.undo_move()
    flip = not flip

def run_mcts(position):
  root = Node()
  evaluate(root, position)

  SIMULATION_NUM = 800

  for _ in range(SIMULATION_NUM):
    node = root
    search_path = collections.deque([node])

    while node.expanded():
      move, node = select_child(node)

      position.make_move(move)

      # record path
      search_path.append(node)

    value = evaluate(node, position)
    backpropagate(position, search_path, value)

def main():
  position = minishogilib.Position()
  position.set_start_position()
  # position.print()
  moves = position.generate_moves(True, True)
  print(len(moves))

  run_mcts(position)

if __name__ == '__main__':
  # output minishogilib version
  print(minishogilib.version())

  # fix the seed
  np.random.seed(0)

  main()
