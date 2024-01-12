use std::{collections::HashMap, fmt::Display};

use crate::parse::{Context, Decl, Node, NodeId, Product, Sum};

pub type StringId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
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
    Constant(StringId),
    Unresolved(StringId),
    Resolved(usize),
    Block(BlockId),
}

impl Operand {
    fn to_index(&self) -> Option<usize> {
        match self {
            Self::Resolved(x) => Some(*x),
            _ => None,
        }
    }
}

pub type BlockId = usize;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    Block,
    As,
    Fn,
    Arg,
    Product,
    Sum,
    Hole,
    Call,
    Match,
    Try,
    Unreachable,
    Int,
}

impl Display for Opcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Block => "BLOCK",
                Self::As => "AS",
                Self::Fn => "FN",
                Self::Arg => "ARG",
                Self::Product => "PRODUCT",
                Self::Sum => "SUM",
                Self::Hole => "HOLE",
                Self::Call => "CALL",
                Self::Match => "MATCH",
                Self::Try => "TRY",
                Self::Unreachable => "UNREACHABLE",
                Self::Int => "INT",
            }
        )
    }
}

#[derive(Debug)]
pub struct Inst {
    op: Opcode,
    args: Vec<Operand>,
}

#[derive(Debug)]
pub enum BlockKind {
    Value,
    Func,
}

#[derive(Debug)]
pub struct Block {
    kind: BlockKind,
    insts: Vec<usize>,
    parent: Option<BlockId>,
    labels: HashMap<String, usize>,
}

#[derive(Default, Debug)]
pub struct IR {
    insts: Vec<Inst>,
    blocks: Vec<Block>,
    strings: Vec<String>,
    curr: BlockId,
}

impl IR {
    pub fn add_block(&mut self, block: Block) -> BlockId {
        self.blocks.push(block);
        self.blocks.len() - 1
    }

    pub fn add_inst(&mut self, inst: Inst) -> Operand {
        self.insts.push(inst);
        let id = self.insts.len() - 1;
        self.blocks[self.curr].insts.push(id);
        Operand::Resolved(id)
    }

    pub fn add_string(&mut self, string: String) -> StringId {
        self.strings.push(string);
        self.strings.len() - 1
    }

    pub fn add_label(&mut self, inst: usize, label: String) {
        self.blocks[self.curr].labels.insert(label, inst);
    }

    fn lookup(&self, block: &Block, label: &str) -> Option<usize> {
        match block.labels.get(label) {
            Some(x) => Some(x.clone()),
            None => match block.parent {
                Some(parent) => self.lookup(&self.blocks[parent], label),
                None => None,
            },
        }
    }

    fn display_operand(&self, arg: Operand) -> String {
        match arg {
            Operand::Type => "type".to_string(),
            Operand::Unreachable => "unreachable".to_string(),
            Operand::U8 => "u8".to_string(),
            Operand::U16 => "u16".to_string(),
            Operand::U32 => "u32".to_string(),
            Operand::U64 => "u64".to_string(),
            Operand::Usize => "usize".to_string(),
            Operand::I8 => "i8".to_string(),
            Operand::I16 => "i16".to_string(),
            Operand::I32 => "i32".to_string(),
            Operand::I64 => "i64".to_string(),
            Operand::Isize => "isize".to_string(),
            Operand::F32 => "f32".to_string(),
            Operand::F64 => "f64".to_string(),
            Operand::Block(block) => self.display_block(&self.blocks[block]),
            Operand::Constant(id) => self.strings[id].clone(),
            Operand::Unresolved(id) => self.strings[id].clone(),
            Operand::Resolved(id) => format!("%{}", id),
        }
    }

    fn display_inst(&self, inst: &Inst) -> String {
        format!(
            "{} {}",
            inst.op,
            inst.args
                .iter()
                .map(|x| self.display_operand(*x))
                .collect::<Vec<String>>()
                .join(" ")
        )
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
                        self.display_inst(&self.insts[*inst])
                    )
                })
                .flat_map(|x| x.split('\n').map(|x| x.to_owned()).collect::<Vec<String>>())
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
            .find(|block| block.insts.contains(&id))
            .unwrap()
    }

    fn _resolve(&self, id: Operand, base: usize) -> Operand {
        match id {
            Operand::Unresolved(x) => Operand::Resolved(
                self.lookup(self.get_block(base), &self.strings[x])
                    .expect(&format!("Failed to resolve {x}")),
            ),
            x => x.clone(),
        }
    }

    fn resolve(&mut self) {
        self.insts = self
            .insts
            .iter()
            .enumerate()
            .map(|(i, inst)| Inst {
                op: inst.op,
                args: inst.args.iter().map(|x| self._resolve(*x, i)).collect(),
            })
            .collect::<Vec<Inst>>()
    }
}

fn node(ir: &mut IR, ctx: &Context, id: NodeId) -> Operand {
    println!("{:?}", ctx.nodes[id]);
    match &ctx.nodes[id] {
        Node::Type => Operand::Type,
        Node::Var(Some(x)) => match x.as_str() {
            "u32" => Operand::U32,
            _ => Operand::Unresolved(ir.add_string(x.clone())),
        },
        Node::Call((func, arg)) => {
            let arg = node(ir, ctx, *arg);
            let func = ir.add_string(func.clone());
            ir.add_inst(Inst {
                op: Opcode::Call,
                args: vec![Operand::Unresolved(func), arg],
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
            let arg = ir.add_inst(Inst {
                op: Opcode::Arg,
                args: Vec::new(),
            });
            ir.add_label(arg.to_index().unwrap(), x.unwrap());
            node(ir, ctx, *body);

            ir.curr = ir.blocks[ir.curr].parent.unwrap();
            ir.add_inst(Inst {
                op: Opcode::Block,
                args: vec![Operand::Block(block)],
            })
        }
        Node::Function((arg, ret)) => {
            let arg = node(ir, ctx, *arg);
            let ret = node(ir, ctx, *ret);
            ir.add_inst(Inst {
                op: Opcode::Fn,
                args: vec![arg, ret],
            })
        }
        Node::Product(Product(fields)) => {
            let fields = fields
                .iter()
                .map(|field| node(ir, ctx, field.1))
                .collect::<Vec<Operand>>();
            ir.add_inst(Inst {
                op: Opcode::Product,
                args: fields,
            })
        }
        Node::Sum(Sum(variants)) => {
            let variants = variants
                .iter()
                .map(|variant| node(ir, ctx, variant.1))
                .collect::<Vec<Operand>>();
            ir.add_inst(Inst {
                op: Opcode::Sum,
                args: variants,
            })
        }
        Node::TypeVar(Some(x)) => {
            let x = ir.add_string(x.clone());
            ir.add_inst(Inst {
                op: Opcode::Hole,
                args: vec![Operand::Unresolved(x)],
            })
        }
        Node::TypeVar(None) => ir.add_inst(Inst {
            op: Opcode::Hole,
            args: Vec::new(),
        }),
        Node::Int(x) => {
            let x = ir.add_string(x.clone());
            ir.add_inst(Inst {
                op: Opcode::Int,
                args: vec![Operand::Constant(x)],
            })
        }
        _ => todo!(),
    }
}

fn decl(ir: &mut IR, ctx: &Context, decl: &Decl, parent: BlockId) -> Operand {
    let block = ir.add_block(Block {
        kind: BlockKind::Value,
        insts: Vec::new(),
        parent: Some(parent),
        labels: HashMap::new(),
    });

    ir.curr = block;
    let ty = node(ir, ctx, decl.ty);
    let val = node(ir, ctx, decl.value);
    ir.add_inst(Inst {
        op: Opcode::As,
        args: vec![ty, val],
    });
    ir.curr = parent;

    let id = ir.add_inst(Inst {
        op: Opcode::Block,
        args: vec![Operand::Block(block)],
    });
    ir.add_label(id.to_index().unwrap(), decl.ident.clone());
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
