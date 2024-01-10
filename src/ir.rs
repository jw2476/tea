use std::fmt::Display;

use crate::parse::{Context, Decl, Node, NodeId};

#[derive(Debug, Clone, Copy)]
pub enum InstId {
    Type,
    Unreachable,
    U8,
    U16,
    U32,
    U64,
    Usize,
    I8,
    I16,
    I32,
    I64,
    Isize,
    F32,
    F64,
    Other(usize),
}

impl InstId {
    fn to_index(self) -> Option<usize> {
        match self {
            Self::Other(x) => Some(x),
            _ => None,
        }
    }
}

impl Display for InstId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Type => "type".to_string(),
                Self::Unreachable => "unreachable".to_string(),
                Self::U8 => "u8".to_string(),
                Self::U16 => "u16".to_string(),
                Self::U32 => "u32".to_string(),
                Self::U64 => "u64".to_string(),
                Self::Usize => "usize".to_string(),
                Self::I8 => "i8".to_string(),
                Self::I16 => "i16".to_string(),
                Self::I32 => "i32".to_string(),
                Self::I64 => "i64".to_string(),
                Self::Isize => "isize".to_string(),
                Self::F32 => "f32".to_string(),
                Self::F64 => "f64".to_string(),
                Self::Other(id) => format!("%{}", id),
            }
        )
    }
}

#[derive(Debug)]
pub enum Identifier {
    Unresolved(String),
    Resolved(InstId),
}

impl Display for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unresolved(x) => write!(f, "{x}"),
            Self::Resolved(x) => write!(f, "{x}"),
        }
    }
}

pub type BlockId = usize;
#[derive(Debug)]
pub enum Inst {
    Block(BlockId),
    As { ty: InstId, val: InstId },
    Fn { arg: InstId, ret: InstId },
    Arg,
    Product(Vec<InstId>),
    Sum(Vec<InstId>),
    Hole(Option<String>),
    Call { func: Identifier, arg: InstId },
    Match { variant: usize, val: InstId },
    Try(Vec<InstId>),
    Unreachable,
    Int(String),
}

#[derive(Debug)]
pub enum BlockKind {
    Value,
    Func,
}

#[derive(Debug)]
pub struct Block {
    kind: BlockKind,
    insts: Vec<InstId>,
    parent: Option<BlockId>,
}

#[derive(Default, Debug)]
pub struct IR {
    insts: Vec<Inst>,
    blocks: Vec<Block>,
    curr: BlockId,
}

impl IR {
    pub fn add_block(&mut self, block: Block) -> BlockId {
        self.blocks.push(block);
        self.blocks.len() - 1
    }

    pub fn add_inst(&mut self, inst: Inst) -> InstId {
        self.insts.push(inst);
        let id = InstId::Other(self.insts.len() - 1);
        self.blocks[self.curr].insts.push(id);
        id
    }

    pub fn next_id(&self) -> InstId {
        InstId::Other(self.insts.len())
    }

    fn display_inst(&self, inst: &Inst) -> String {
        match inst {
            Inst::As { ty, val } => format!("AS {ty} {val}"),
            Inst::Block(block) => self.display_block(&self.blocks[*block]),
            Inst::Int(x) => format!("INT {x}"),
            Inst::Call { func, arg } => format!("CALL {func} {arg}"),
            Inst::Hole(ident) => format!("HOLE {}", ident.clone().unwrap_or_default()),
            _ => todo!(),
        }
    }

    fn display_block(&self, block: &Block) -> String {
        format!(
            "block {{\n{}\n}}",
            block
                .insts
                .iter()
                .map(|inst| format!(
                    "{inst} = {}",
                    self.display_inst(&self.insts[inst.to_index().unwrap()])
                ))
                .collect::<Vec<String>>()
                .join("\n")
                .split("\n")
                .map(|x| format!("  {x}"))
                .collect::<Vec<String>>()
                .join("\n")
        )
    }

    pub fn display(&self) -> String {
        self.blocks
            .iter()
            .filter(|block| block.parent.is_none())
            .map(|block| self.display_block(block))
            .collect::<Vec<String>>()
            .join("\n")
    }
}

fn node(ir: &mut IR, ctx: &Context, id: NodeId) -> InstId {
    match &ctx.nodes[id] {
        Node::Type => InstId::Type,
        Node::Var(Some(x)) => match x.as_str() {
            "u32" => InstId::U32,
            _ => todo!(),
        },
        Node::Call((func, arg)) => {
            let arg = node(ir, ctx, *arg);
            ir.add_inst(Inst::Call {
                func: Identifier::Unresolved(func.clone()),
                arg,
            })
        }
        Node::TypeVar(x) => ir.add_inst(Inst::Hole(x.clone())),
        Node::Int(x) => ir.add_inst(Inst::Int(x.clone())),
        _ => todo!(),
    }
}

fn decl(ir: &mut IR, ctx: &Context, decl: &Decl, parent: BlockId) -> InstId {
    let block = ir.add_block(Block {
        kind: BlockKind::Value,
        insts: Vec::new(),
        parent: Some(parent),
    });

    ir.curr = block;
    let ty = node(ir, ctx, decl.ty);
    let val = node(ir, ctx, decl.value);
    ir.add_inst(Inst::As { ty, val });
    ir.curr = parent;

    ir.add_inst(Inst::Block(block))
}

pub fn generate(ctx: &Context, decls: Vec<Decl>) -> IR {
    let mut ir = IR::default();
    ir.add_block(Block {
        kind: BlockKind::Value,
        insts: Vec::new(),
        parent: None,
    });

    decls.iter().for_each(|x| {
        decl(&mut ir, ctx, x, 0);
    });
    ir
}
