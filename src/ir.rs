use std::{collections::HashMap, fmt::Display};

use crate::parse::{Context, Decl, Node, NodeId, Product, Sum};

#[derive(Debug, Clone, PartialEq, Eq)]
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
    Unresolved(String),
    Resolved(usize),
}

impl InstId {
    fn to_index(&self) -> Option<usize> {
        match self {
            Self::Resolved(x) => Some(*x),
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
                Self::Unresolved(id) => format!("%{}", id),
                Self::Resolved(id) => format!("%{}", id),
            }
        )
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
    Call { func: InstId, arg: InstId },
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
    labels: HashMap<String, InstId>,
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
        let id = InstId::Resolved(self.insts.len() - 1);
        self.blocks[self.curr].insts.push(id.clone());
        id
    }

    pub fn add_label(&mut self, inst: InstId, label: String) {
        self.blocks[self.curr].labels.insert(label, inst);
    }

    fn lookup(&self, block: &Block, label: &str) -> Option<InstId> {
        match block.labels.get(label) {
            Some(x) => Some(x.clone()),
            None => match block.parent {
                Some(parent) => self.lookup(&self.blocks[parent], label),
                None => None,
            },
        }
    }

    fn display_inst(&self, inst: &Inst) -> String {
        match inst {
            Inst::As { ty, val } => format!("AS {ty} {val}"),
            Inst::Block(block) => self.display_block(&self.blocks[*block]),
            Inst::Int(x) => format!("INT {x}"),
            Inst::Call { func, arg } => format!("CALL {func} {arg}"),
            Inst::Hole(ident) => format!("HOLE {}", ident.clone().unwrap_or_default()),
            Inst::Fn { arg, ret } => {
                format!("FN {arg} {ret}")
            }
            Inst::Arg => "ARG".to_string(),
            Inst::Product(fields) => format!(
                "PRODUCT {}",
                fields
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
                    .join(" ")
            ),
            Inst::Sum(variants) => format!(
                "SUM {}",
                variants
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
                    .join(" ")
            ),
            _ => {
                println!("{inst:?}");
                todo!()
            }
        }
    }

    fn display_block(&self, block: &Block) -> String {
        format!(
            "block {{\n{}\n}}",
            block
                .insts
                .iter()
                .map(|inst| {
                    let label = block.labels.iter().find(|(_, i)| *i == inst);
                    format!(
                        "{}{inst} = {}",
                        label.map(|(x, _)| format!("{x}: ")).unwrap_or_default(),
                        self.display_inst(&self.insts[inst.to_index().unwrap()])
                    )
                })
                .collect::<Vec<String>>()
                .join("\n")
                .split('\n')
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

    fn get_block(&self, id: usize) -> &Block {
        self.blocks
            .iter()
            .find(|block| {
                block
                    .insts
                    .iter()
                    .find(|inst| inst == &&InstId::Resolved(id))
                    .is_some()
            })
            .unwrap()
    }

    fn _resolve(&self, id: &InstId, base: usize) -> InstId {
        match id {
            InstId::Unresolved(x) => self
                .lookup(self.get_block(base), x)
                .expect(&format!("Failed to resolve {x}")),
            x => x.clone(),
        }
    }

    fn resolve(&mut self) {
        self.insts = self
            .insts
            .iter()
            .enumerate()
            .map(|(i, inst)| match inst {
                Inst::As { ty, val } => Inst::As {
                    ty: self._resolve(ty, i),
                    val: self._resolve(val, i),
                },
                Inst::Block(block) => Inst::Block(*block),
                Inst::Fn { arg, ret } => Inst::Fn {
                    arg: self._resolve(arg, i),
                    ret: self._resolve(ret, i),
                },
                Inst::Arg => Inst::Arg,
                Inst::Product(fields) => {
                    Inst::Product(fields.iter().map(|field| self._resolve(field, i)).collect())
                }
                Inst::Sum(variants) => Inst::Sum(
                    variants
                        .iter()
                        .map(|variant| self._resolve(variant, i))
                        .collect(),
                ),
                Inst::Hole(_) => todo!(),
                Inst::Call { func, arg } => Inst::Call {
                    func: self._resolve(func, i),
                    arg: self._resolve(arg, i),
                },
                Inst::Match { variant, val } => todo!(),
                Inst::Try(branches) => {
                    Inst::Try(branches.iter().map(|x| self._resolve(x, i)).collect())
                }
                Inst::Unreachable => Inst::Unreachable,
                Inst::Int(x) => Inst::Int(x.clone()),
            })
            .collect::<Vec<Inst>>()
    }
}

fn node(ir: &mut IR, ctx: &Context, id: NodeId) -> InstId {
    println!("{:?}", ctx.nodes[id]);
    match &ctx.nodes[id] {
        Node::Type => InstId::Type,
        Node::Var(Some(x)) => match x.as_str() {
            "u32" => InstId::U32,
            _ => InstId::Unresolved(x.clone()),
        },
        Node::Call((func, arg)) => {
            let arg = node(ir, ctx, *arg);
            ir.add_inst(Inst::Call {
                func: InstId::Unresolved(func.clone()),
                arg,
            })
        }
        Node::Lambda((arg, body)) => {
            let Node::Var(x) = ctx.nodes[*arg].clone() else {
                todo!()
            };
            let block = ir.add_block(Block {
                kind: BlockKind::Func,
                insts: Vec::new(),
                parent: Some(ir.curr),
                labels: HashMap::new(),
            });
            ir.curr = block;
            let arg = ir.add_inst(Inst::Arg);
            ir.add_label(arg, x.unwrap());
            node(ir, ctx, *body);

            ir.curr = ir.blocks[ir.curr].parent.unwrap();
            ir.add_inst(Inst::Block(block))
        }
        Node::Function((arg, ret)) => {
            let arg = node(ir, ctx, *arg);
            let ret = node(ir, ctx, *ret);
            ir.add_inst(Inst::Fn { arg, ret })
        }
        Node::Product(Product(fields)) => {
            let fields = fields
                .iter()
                .map(|field| node(ir, ctx, field.1))
                .collect::<Vec<InstId>>();
            ir.add_inst(Inst::Product(fields))
        }
        Node::Sum(Sum(variants)) => {
            let variants = variants
                .iter()
                .map(|variant| node(ir, ctx, variant.1))
                .collect::<Vec<InstId>>();
            ir.add_inst(Inst::Sum(variants))
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
        labels: HashMap::new(),
    });

    ir.curr = block;
    let ty = node(ir, ctx, decl.ty);
    let val = node(ir, ctx, decl.value);
    ir.add_inst(Inst::As { ty, val });
    ir.curr = parent;

    let id = ir.add_inst(Inst::Block(block));
    ir.add_label(id.clone(), decl.ident.clone());
    id
}

pub fn generate(ctx: &Context, decls: Vec<Decl>) -> IR {
    let mut ir = IR::default();
    ir.add_block(Block {
        kind: BlockKind::Value,
        insts: Vec::new(),
        parent: None,
        labels: HashMap::new(),
    });

    decls.iter().for_each(|x| {
        decl(&mut ir, ctx, x, 0);
    });
    ir.resolve();
    ir
}
