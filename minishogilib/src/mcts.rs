use pyo3::prelude::*;

use r#move::*;
use types::*;

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
    pub fn new(parent: usize, policy: f32) -> Node {
        Node {
            n: 0,
            v: 0.0,
            p: policy,
            w: 0.0,
            m: NULL_MOVE,
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
    pub game_tree: [Node; 100000]  // 0 represents nothing. Indexing should be start from 1.
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
}
