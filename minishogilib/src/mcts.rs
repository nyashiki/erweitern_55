use r#move::*;
use position::*;
use types::*;

use pyo3::prelude::*;
use numpy::PyArray1;

pub struct Node {
    pub n: u32,
    pub v: f32,
    pub p: f32,
    pub w: f32,
    pub m: Move,
    pub parent: usize,
    pub children: std::vec::Vec<usize>,
    pub is_terminal: bool,
    pub virtual_loss: f32
}

impl Node {
    pub fn new(parent: usize, m: Move, policy: f32) -> Node {
        Node {
            n: 0,
            v: 0.0,
            p: policy,
            w: 0.0,
            m: m,
            parent: parent,
            children: Vec::new(),
            is_terminal: false,
            virtual_loss: 0.0
        }
    }

    pub fn get_puct(&self, parent_n: u32) -> f32 {
        const C_BASE: f32 = 19652.0;
        const C_INIT: f32 = 1.25;

        let c: f32 = ((1.0 + (self.n as f32) + C_BASE) / C_BASE).log2() + C_INIT;
        let q: f32 = if self.n == 0 { 0.0 } else { (1.0 - self.w / ((self.n as f32 + self.virtual_loss))) };
        let u: f32 = c * self.p * (parent_n as f32).sqrt() / (1.0 + (self.n as f32) + self.virtual_loss);

        return q + u;
    }

    pub fn expanded(&self) -> bool {
        return self.children.len() > 0 && !self.is_terminal;
    }
}

#[pyclass]
pub struct MCTS {
    pub game_tree: [Node; 100000], // 0 represents nothing. Indexing should be start from 1.
    pub node_count: usize
}

impl MCTS {
    pub fn select_puct_max_child(&self, node: usize) -> usize {
        let mut puct_max: f32 = 0.0;
        let mut puct_max_child: usize = 0;

        for child in &self.game_tree[node].children {
            let puct = self.game_tree[*child].get_puct(self.game_tree[node].n);

            if puct > puct_max {
                puct_max = puct;
                puct_max_child = *child;
            }
        }

        return puct_max_child;
    }

    pub fn select_leaf(&mut self, root_node: usize, position: &mut Position) -> usize {
        let mut node = root_node;

        while self.game_tree[node].expanded() {
            self.game_tree[node].virtual_loss += 1.0;
            node = self.select_puct_max_child(node);
            position.do_move(&self.game_tree[node].m);
        }

        return node;
    }

    pub fn evaluate(&mut self, node: usize, position: &Position, np_policy: PyArray1<f32>, mut value: f32) -> f32 {
        let policy = np_policy.as_slice();
        let mut legal_policy_sum: f32 = 0.0;

        let moves = position.generate_moves();

        for m in &moves {
            let index = move_to_policy_index(m, position.side_to_move);
            legal_policy_sum += policy[index];
        }

        let (is_repetition, is_check_repetition) = position.is_repetition();

        if is_repetition || moves.len() == 0 {
            self.game_tree[node].is_terminal = true;
        }

        // win or lose is determined by the game rule
        if self.game_tree[node].is_terminal {
            if is_check_repetition {
                value = 0.0;
            } else if is_repetition {
                value = if position.side_to_move == Color::White { 0.0 } else { 1.0 }
            } else {
                value = 0.0
            }
        }

        // set policy and vaue
        for m in &moves {
            let index = move_to_policy_index(m, position.side_to_move);

            self.game_tree[self.node_count] = Node::new(node, *m, policy[index] / legal_policy_sum);
            self.game_tree[node].children.push(self.node_count);
            self.node_count += 1;
        }
        self.game_tree[node].v = value;

        return value;
    }

    pub fn backpropagate(&mut self, leaf_node: usize, value: f32) {
        let mut node = leaf_node;
        let mut flip = false;

        while node != 0 {
            self.game_tree[node].w += if !flip { value } else { 1.0 - value };
            self.game_tree[node].n += 1;
            self.game_tree[node].virtual_loss -= 1.0;

            node = self.game_tree[node].parent;
            flip = !flip;
        }
    }
}

fn move_to_policy_index(m: &Move, c: Color) -> usize {
    let index = if m.amount == 0 {
                        if c == Color::White {
                            (64 + m.get_hand_index(), m.to / 5, m.to % 5)
                        } else {
                            (64 + m.get_hand_index(), 4 - m.to / 5, 4 - m.to % 5)
                        }
                    } else {
                        if m.get_promotion() {
                            if c == Color::White {
                                (32 + 4 * m.direction as usize + m.amount - 1, m.from / 5, m.from % 5)
                            } else {
                                (32 + 4 * m.direction as usize + m.amount - 1, 4 - m.from / 5, 4 - m.from % 5)
                            }
                        } else {
                            if c == Color::White {
                                (4 * m.direction as usize + m.amount - 1, m.from / 5, m.from % 5)
                            } else {
                                (4 * m.direction as usize + m.amount - 1, 4 - m.from / 5, 4 - m.from % 5)
                            }
                        }
                    };

    return index.0 * 25 + index.1 * 5 + index.2;
}
