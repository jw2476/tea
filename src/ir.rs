use crate::parse::{Context, Decl, Node, NodeId, Product, Sum};

pub type GraphId = usize;
pub type StringId = usize;
pub type NodeId = usize;
pub type TypeId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Constant {
    Int(StringId),
    Ident(StringId),
    Graph(GraphId),
}

pub enum NodeKind {
    Input,
    Output,
    Get,
    Product,
    Match,
    Variant,
    Call,
    Constant(Constant),
    Stop,
}

pub struct Node {
    kind: NodeKind,
    inputs: Vec<NodeId>,
    ty: TypeId,
}

pub struct Graph {
    nodes: Vec<NodeId>,


pub struct IR {
    graphs: Vec<Graph>,
    strings: Vec<String>,
    nodes: Vec<Node>,
    types: Vec<Type>,
}

pub fn generate(ctx: &Context, decls: Vec<Decl>) -> IR {
    let mut ir = IR::default();

    decls.iter().for_each(|x| {
        decl(&mut ir, ctx, x, 0);
    });
    ir
}
