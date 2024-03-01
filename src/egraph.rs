use std::{collections::HashMap, hash::Hash};

use crate::ast::{ExprId, StringId, TypeId, AST, Expr};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClassId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Get(usize),
    Int(StringId),
    Arg,
    Return,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    op: Op,
    args: Vec<ClassId>,
}

#[derive(Clone, Debug, Default)]
pub struct Class {
    nodes: Vec<Node>,
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    classes: Vec<Class>,
}

pub struct ScopedMap<K, V> {
    maps: Vec<HashMap<K, V>>,
}

impl<K: Eq + Hash, V> ScopedMap<K, V> {
    pub fn new() -> Self {
        Self { maps: Vec::new() }
    }

    pub fn push(&mut self) {
        self.maps.push(HashMap::new())
    }

    pub fn pop(&mut self) {
        self.maps.pop();
    }

    pub fn insert(&mut self, key: K, value: V) {
        self.maps.last_mut().unwrap().insert(key, value);
    }

    pub fn get(&self, k: &K) -> Option<&V> {
        self.maps.iter().rev().find_map(|map| map.get(k))
    }
}

fn expr_to_node(
    graph: &mut Graph,
    ast: &AST,
    expr: ExprId,
    types: &[TypeId],
    decls: &mut ScopedMap<StringId, ExprId>,
) -> ClassId {
    let node = match ast[expr].clone() {
        Expr::Arg => Node { op: Op::Arg, args: Vec::new() },
        Expr::Int(int) => Node { op: Op::Int(int), args: Vec::new() },
        Expr::Call(ident, arg) => 
    }
}

pub fn to_egraph(ast: &AST, types: &[TypeId]) -> Graph {
    Graph::default()
}
