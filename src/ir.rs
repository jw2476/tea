use std::{
    collections::HashMap,
    ops::{Index, IndexMut},
};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StringId(usize);
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TypeId(usize);
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockId(usize);
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct OpId(usize);
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RegionId(usize);

#[derive(Clone, Copy, Hash, Debug, PartialEq, Eq)]
pub enum Attr {
    Type(TypeId),
    Symbol(StringId),
    Region(RegionId),
    Int(StringId),
}

impl Attr {
    pub fn ty(&self) -> Option<TypeId> {
        match self {
            Self::Type(id) => Some(*id),
            _ => None,
        }
    }

    pub fn symbol(&self) -> Option<StringId> {
        match self {
            Self::Symbol(id) => Some(*id),
            _ => None,
        }
    }

    pub fn region(&self) -> Option<RegionId> {
        match self {
            Self::Region(id) => Some(*id),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Hash, Debug, PartialEq, Eq)]
pub enum Value {
    Op(OpId),
    Arg(BlockId),
}

impl From<OpId> for Value {
    fn from(value: OpId) -> Self {
        Self::Op(value)
    }
}

impl Value {
    pub fn op(&self) -> Option<OpId> {
        match self {
            Self::Op(id) => Some(*id),
            _ => None,
        }
    }

    pub fn arg(&self) -> Option<BlockId> {
        match self {
            Self::Arg(id) => Some(*id),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Op {
    id: StringId,
    args: Vec<Value>,
    attrs: HashMap<StringId, Attr>,
    ty: Option<TypeId>,
    stored: bool,
}

#[derive(Clone, Debug)]
pub struct Block {
    label: StringId,
    arg: Option<TypeId>,
    ops: Vec<OpId>,
}

#[derive(Clone, Debug)]
pub struct Region {
    blocks: Vec<BlockId>,
}

#[derive(Clone, Debug)]
pub enum Type {
    Product(Vec<TypeId>),
    Sum(Vec<TypeId>),
    Function(TypeId, TypeId),
    U8,
    U16,
    U32,
    U64,
    USize,
    I8,
    I16,
    I32,
    I64,
    ISize,
    F32,
    F64,
}

impl Type {
    pub fn product(&self) -> Option<&[TypeId]> {
        match self {
            Self::Product(fields) => Some(fields),
            _ => None,
        }
    }
    pub fn sum(&self) -> Option<&[TypeId]> {
        match self {
            Self::Sum(variants) => Some(variants),
            _ => None,
        }
    }

    pub fn function(&self) -> Option<(TypeId, TypeId)> {
        match self {
            Self::Function(arg, ret) => Some((*arg, *ret)),
            _ => None,
        }
    }

    pub fn is_primitive(&self) -> bool {
        !matches!(self, Self::Product(_) | Self::Sum(_) | Self::Function(_, _))
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub enum Decl {
    Op(OpId),
    TypeAlias(StringId, TypeId),
}

impl Decl {
    pub fn op(&self) -> Option<OpId> {
        match self {
            Self::Op(id) => Some(*id),
            _ => None,
        }
    }
    pub fn type_alias(&self) -> Option<(StringId, TypeId)> {
        match self {
            Self::TypeAlias(str, ty) => Some((*str, *ty)),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct IR {
    decls: Vec<Decl>,
    types: Vec<Type>,
    strings: Vec<String>,
    ops: Vec<Op>,
    blocks: Vec<Block>,
    regions: Vec<Region>,
}

impl IR {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_string<T: ToString>(&mut self, string: T) -> StringId {
        let string = string.to_string();
        if let Some(index) = self.strings.iter().position(|s| s == &string) {
            StringId(index)
        } else {
            self.strings.push(string);
            StringId(self.strings.len() - 1)
        }
    }

    fn add_op(&mut self, op: Op) -> OpId {
        self.ops.push(op);
        OpId(self.ops.len() - 1)
    }

    pub fn add_ty(&mut self, ty: Type) -> TypeId {
        self.types.push(ty);
        TypeId(self.types.len() - 1)
    }

    fn add_block(&mut self, block: Block) -> BlockId {
        self.blocks.push(block);
        BlockId(self.blocks.len() - 1)
    }

    fn add_region(&mut self, region: Region) -> RegionId {
        self.regions.push(region);
        RegionId(self.regions.len() - 1)
    }

    pub fn append_op(&mut self, block: BlockId, op: Op) -> OpId {
        let id = self.add_op(op);
        self[block].ops.push(id);
        id
    }

    pub fn new_region(&mut self) -> RegionId {
        let region = Region { blocks: Vec::new() };
        self.add_region(region)
    }

    pub fn append_block(
        &mut self,
        region: RegionId,
        label: StringId,
        arg: Option<TypeId>,
    ) -> BlockId {
        let block = Block {
            label,
            arg,
            ops: Vec::new(),
        };
        let id = self.add_block(block);
        self[region].blocks.push(id);
        id
    }

    fn display_block(&self, block: BlockId) -> String {
        let block = &self[block];
        let arg = block
            .arg
            .map(|x| self.display_ty(x))
            .unwrap_or_else(|| " <infer>".to_string());
        format!(
            "^{}{}:\n{}\n",
            self[block.label],
            arg,
            block
                .ops
                .iter()
                .map(|op| self.display_op(*op))
                .collect::<Vec<String>>()
                .join("\n")
                .split("\n")
                .map(|x| format!("    {x}"))
                .collect::<Vec<String>>()
                .join("\n")
        )
    }

    fn display_region(&self, region: RegionId) -> String {
        let region = &self[region];
        format!(
            "{{\n{}\n}}",
            region
                .blocks
                .iter()
                .map(|block| self.display_block(*block))
                .collect::<Vec<String>>()
                .join("\n")
        )
    }

    fn display_attr(&self, attr: Attr) -> String {
        match attr {
            Attr::Type(ty) => self.display_ty(ty),
            Attr::Symbol(str) => self[str].clone(),
            Attr::Region(region) => self.display_region(region),
            Attr::Int(str) => self[str].clone(),
        }
    }

    fn display_op(&self, id: OpId) -> String {
        let op = &self[id];
        let result = if op.stored {
            format!("%{} = ", id.0)
        } else {
            String::new()
        };
        let args = op
            .args
            .iter()
            .map(|arg| match arg {
                Value::Op(id) => format!("%{}", id.0),
                Value::Arg(id) => todo!(),
            })
            .collect::<Vec<String>>()
            .join(", ");
        let attrs = op
            .attrs
            .iter()
            .map(|(k, v)| format!("{}: {}", self[*k], self.display_attr(*v)))
            .collect::<Vec<String>>()
            .join(", ");
        let ty = format!(
            " : {}",
            op.ty
                .map(|x| self.display_ty(x))
                .unwrap_or_else(|| "<infer>".to_string())
        );
        format!("{}{}({}) {{{}}}{}", result, self[op.id], args, attrs, ty)
    }

    fn display_ty(&self, ty: TypeId) -> String {
        match &self[ty] {
            Type::Product(fields) => format!(
                "({})",
                fields
                    .iter()
                    .map(|x| self.display_ty(*x))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            Type::Sum(variants) => variants
                .iter()
                .map(|x| self.display_ty(*x))
                .collect::<Vec<String>>()
                .join(" | "),
            Type::Function(arg, ret) => {
                format!("{} -> {}", self.display_ty(*arg), self.display_ty(*ret))
            }
            Type::U8 => "u8".to_string(),
            Type::U16 => "u16".to_string(),
            Type::U32 => "u32".to_string(),
            Type::U64 => "u64".to_string(),
            Type::USize => "usize".to_string(),
            Type::I8 => "i8".to_string(),
            Type::I16 => "i16".to_string(),
            Type::I32 => "i32".to_string(),
            Type::I64 => "i64".to_string(),
            Type::ISize => "isize".to_string(),
            Type::F32 => "f32".to_string(),
            Type::F64 => "f64".to_string(),
        }
    }

    pub fn display(&self) -> String {
        self.decls
            .iter()
            .map(|decl| match decl {
                Decl::Op(id) => self.display_op(*id),
                Decl::TypeAlias(alias, ty) => {
                    format!("!{} = {}", self[*alias], self.display_ty(*ty))
                }
            })
            .collect::<Vec<String>>()
            .join("\n")
    }
}

macro_rules! impl_ir_index {
    ($id:ty, $t:ty, $i:ident) => {
        impl Index<$id> for IR {
            type Output = $t;

            fn index(&self, index: $id) -> &Self::Output {
                &self.$i[index.0]
            }
        }

        impl IndexMut<$id> for IR {
            fn index_mut(&mut self, index: $id) -> &mut Self::Output {
                &mut self.$i[index.0]
            }
        }
    };
}

impl_ir_index!(TypeId, Type, types);
impl_ir_index!(StringId, String, strings);
impl_ir_index!(OpId, Op, ops);
impl_ir_index!(BlockId, Block, blocks);
impl_ir_index!(RegionId, Region, regions);

pub mod arith {
    use super::*;

    pub fn addi(ir: &mut IR, block: BlockId, lhs: Value, rhs: Value, ty: Option<TypeId>) -> OpId {
        let id = ir.add_string("arith.addi");
        let op = Op {
            id,
            args: vec![lhs, rhs],
            attrs: HashMap::new(),
            ty,
            stored: true,
        };
        ir.append_op(block, op)
    }

    pub fn int(ir: &mut IR, block: BlockId, int: StringId, ty: Option<TypeId>) -> OpId {
        let id = ir.add_string("arith.int");
        let value = ir.add_string("value");
        let op = Op {
            id,
            args: Vec::new(),
            attrs: HashMap::from([(value, Attr::Int(int))]),
            ty,
            stored: true,
        };
        ir.append_op(block, op)
    }
}

pub mod func {
    use super::*;

    pub fn func(
        ir: &mut IR,
        block: Option<BlockId>,
        symbol: StringId,
        body: RegionId,
        ty: Option<TypeId>,
    ) -> OpId {
        let id = ir.add_string("func.func");
        let ksymbol = ir.add_string("symbol");
        let kbody = ir.add_string("body");
        let op = Op {
            id,
            args: Vec::new(),
            attrs: HashMap::from([(ksymbol, Attr::Symbol(symbol)), (kbody, Attr::Region(body))]),
            ty,
            stored: false,
        };
        if let Some(block) = block {
            ir.append_op(block, op)
        } else {
            let id = ir.add_op(op);
            ir.decls.push(Decl::Op(id));
            id
        }
    }

    pub fn ret(ir: &mut IR, block: BlockId, value: Value, ty: Option<TypeId>) -> OpId {
        let id = ir.add_string("func.ret");
        let op = Op {
            id,
            args: vec![value],
            attrs: HashMap::new(),
            ty,
            stored: false,
        };
        ir.append_op(block, op)
    }
}
