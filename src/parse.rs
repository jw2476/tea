use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::{Deref, Index, Range, RangeBounds, RangeFrom},
    rc::Rc,
    string::ToString,
};

use nom::{
    branch::alt,
    combinator::opt,
    error::ParseError,
    multi::{many0, many1, separated_list0, separated_list1},
    sequence::delimited,
    Err, IResult, Parser,
};

use crate::lex::Token;

#[derive(Debug)]
pub enum ErrorKind {
    Nom(nom::error::ErrorKind),
    MissingIdent,
    MissingDoubleColon,
    MissingComma,
    MissingLeftSquare,
    MissingRightSquare,
    MissingLeftCurly,
    MissingRightCurly,
    MissingLeftRound,
    MissingRightRound,
    MissingLeftAngle,
    MissingRightAngle,
    MissingArrow,
    MissingSemicolon,
    MissingInteger,
    MissingDecimal,
    MissingEquals,
    MissingPipe,
    MissingKeyword(String),
    MissingWideArrow,
    MissingDot,
    MissingColon,
    MissingUnderscore,
    MissingTick,
}

pub struct Error {
    kind: ErrorKind,
    next: Option<Box<Self>>,
}

impl Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Error")
            .field("kind", &self.kind)
            .field("next", &self.next)
            .finish()
    }
}

impl<'a> ParseError<Input<'a>> for Error {
    fn from_error_kind(input: Input<'a>, kind: nom::error::ErrorKind) -> Self {
        Self {
            kind: ErrorKind::Nom(kind),
            next: None,
        }
    }

    fn append(input: Input<'a>, kind: nom::error::ErrorKind, other: Self) -> Self {
        Self {
            kind: ErrorKind::Nom(kind),
            next: Some(Box::new(other)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InstId(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StringId(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TypeId(pub usize);

impl TypeId {
    pub const UNIT: Self = TypeId(0);
}

#[derive(Clone, Copy, Debug)]
pub enum Opcode {
    Product,
    Variant,
    Get,
    Call,
    Branch,
    Match,
    Int,
    As,
    Field,
    Return,
}

#[derive(Clone, Copy, Debug)]
pub enum Operand {
    Value(InstId),
    Ident(StringId),
    Arg,
    Index(usize),
    Block(BlockId),
    Type(TypeId),
}

impl Operand {
    pub fn to_value(self) -> Option<InstId> {
        match self {
            Self::Value(x) => Some(x),
            _ => None,
        }
    }

    pub fn to_block(self) -> Option<BlockId> {
        match self {
            Self::Block(x) => Some(x),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Inst {
    pub op: Opcode,
    pub args: Vec<Operand>,
}

#[derive(Clone, Debug, Copy)]
pub struct Symbol {
    pub ty: Option<TypeId>,
    pub value: Option<Operand>,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub arg: Option<TypeId>,
    pub insts: Vec<InstId>,
    pub symbols: HashMap<StringId, Symbol>,
    pub blocks: Vec<BlockId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Product(Vec<(StringId, TypeId)>),
    Sum(Vec<(StringId, TypeId)>),
    Function((TypeId, TypeId)),
    Symbol(StringId),
}

#[derive(Clone, Debug)]
pub struct Context {
    pub insts: Vec<Inst>,
    pub strings: Vec<String>,
    pub blocks: Vec<Block>,
    pub types: Vec<Type>,
    pub block: BlockId,
}

impl Context {
    pub fn new() -> Self {
        Self {
            insts: Vec::new(),
            strings: Vec::new(),
            blocks: vec![Block {
                arg: Some(TypeId(0)),
                insts: Vec::new(),
                symbols: HashMap::new(),
                blocks: Vec::new(),
            }],
            types: vec![Type::Product(Vec::new())],
            block: BlockId(0),
        }
    }

    pub fn add_inst(&mut self, inst: Inst) -> InstId {
        self.insts.push(inst);
        self.blocks[self.block.0]
            .insts
            .push(InstId(self.insts.len() - 1));
        InstId(self.insts.len() - 1)
    }

    pub fn add_ty(&mut self, ty: Type) -> TypeId {
        if let Some(index) = self.types.iter().position(|x| x == &ty) {
            TypeId(index)
        } else {
            self.types.push(ty);
            TypeId(self.types.len() - 1)
        }
    }

    pub fn add_block(&mut self, arg: Option<TypeId>) -> BlockId {
        let block = Block {
            arg,
            insts: Vec::new(),
            symbols: HashMap::new(),
            blocks: Vec::new(),
        };
        self.blocks.push(block);
        let id = BlockId(self.blocks.len() - 1);
        self.blocks[self.block.0].blocks.push(id);
        id
    }

    pub fn add_string(&mut self, s: &str) -> StringId {
        if let Some(index) = self.strings.iter().position(|x| x == &s) {
            StringId(index)
        } else {
            self.strings.push(s.to_owned());
            StringId(self.strings.len() - 1)
        }
    }

    pub fn block(&self) -> &Block {
        &self.blocks[self.block.0]
    }

    pub fn block_mut(&mut self) -> &mut Block {
        &mut self.blocks[self.block.0]
    }

    fn display_operand(&self, oper: Operand) -> String {
        match oper {
            Operand::Type(ty) => todo!(),
            Operand::Arg => "$".to_string(),
            Operand::Value(id) => format!("%{}", id.0),
            Operand::Ident(s) => self.strings[s.0].to_owned(),
            Operand::Index(x) => format!("#{x}"),
            Operand::Block(id) => format!(":{}", id.0),
        }
    }

    fn display_ty(&self, ty: TypeId) -> String {
        match &self.types[ty.0] {
            Type::Product(fields) => {
                format!(
                    "{{{}}}",
                    fields
                        .iter()
                        .map(|(label, ty)| format!(
                            "{} {}",
                            self.strings[label.0],
                            self.display_ty(*ty)
                        ))
                        .collect::<Vec<String>>()
                        .join(", ")
                )
            }
            Type::Sum(fields) => {
                format!(
                    "[{}]",
                    fields
                        .iter()
                        .map(|(label, ty)| format!(
                            "{} {}",
                            self.strings[label.0],
                            self.display_ty(*ty)
                        ))
                        .collect::<Vec<String>>()
                        .join(", ")
                )
            }
            Type::Function((arg, ret)) => {
                format!("{} -> {}", self.display_ty(*arg), self.display_ty(*ret))
            }
            Type::Symbol(s) => self.strings[s.0].clone(),
        }
    }

    fn display_inst(&self, inst: InstId) -> String {
        let symbol = self
            .blocks
            .iter()
            .flat_map(|block| block.symbols.iter().collect::<Vec<(&StringId, &Symbol)>>())
            .find(|(_, symbol)| {
                symbol
                    .value
                    .map(|oper| match oper {
                        Operand::Value(x) => inst == x,
                        _ => false,
                    })
                    .unwrap_or_default()
            });

        let symbol = symbol
            .map(|(label, symbol)| {
                format!(
                    "// {}: {}",
                    self.strings[label.0],
                    symbol
                        .ty
                        .map(|x| self.display_ty(x))
                        .unwrap_or_else(|| "inferred".to_owned())
                )
            })
            .unwrap_or_default();

        format!(
            "%{} = {} {}\t{}",
            inst.0,
            format!("{:?}", self.insts[inst.0].op).to_uppercase(),
            self.insts[inst.0]
                .args
                .iter()
                .map(|x| self.display_operand(*x))
                .collect::<Vec<String>>()
                .join(" "),
            symbol
        )
    }

    fn display_block(&self, id: BlockId) -> String {
        let symbol = self
            .blocks
            .iter()
            .flat_map(|block| block.symbols.iter().collect::<Vec<(&StringId, &Symbol)>>())
            .find(|(_, symbol)| {
                symbol
                    .value
                    .map(|oper| match oper {
                        Operand::Block(x) => id == x,
                        _ => false,
                    })
                    .unwrap_or_default()
            });

        let symbol = symbol
            .map(|(label, symbol)| {
                format!(
                    "// {}: {}",
                    self.strings[label.0],
                    symbol
                        .ty
                        .map(|x| self.display_ty(x))
                        .unwrap_or_else(|| "inferred".to_owned())
                )
            })
            .unwrap_or_default();

        let block = &self.blocks[id.0];
        let insts = block
            .insts
            .iter()
            .map(|inst| self.display_inst(*inst))
            .collect::<Vec<String>>()
            .join("\n");
        let blocks = block
            .blocks
            .iter()
            .map(|block| self.display_block(*block))
            .collect::<Vec<String>>()
            .join("\n");
        format!(
            ":{}({}) = {{\t{}\n{}\n}}",
            id.0,
            block
                .arg
                .map(|x| self.display_ty(x))
                .unwrap_or_else(|| "inferred".to_string()),
            symbol,
            (insts + "\n" + &blocks)
                .split('\n')
                .filter(|x| !x.chars().all(|x| x.is_whitespace()))
                .map(|x| format!("    {x}"))
                .collect::<Vec<String>>()
                .join("\n"),
        )
    }

    pub fn display(&self) -> String {
        self.display_block(BlockId(0))
    }
}

#[derive(Clone)]
pub struct Input<'a> {
    ctx: Context,
    tokens: &'a [Token],
}

impl<'a> Input<'a> {
    pub fn slice(self, range: RangeFrom<usize>) -> Self {
        Self {
            ctx: self.ctx,
            tokens: &self.tokens[range],
        }
    }
}

impl<'a> nom::InputLength for Input<'a> {
    fn input_len(&self) -> usize {
        self.tokens.len()
    }
}

pub type Parsed<'a, T> = IResult<Input<'a>, T, Error>;

macro_rules! tok_parser_body {
    ($n:ident, $i:ident, $e:ident) => {
        fn $n(mut input: Input) -> Parsed<StringId> {
            match input.tokens.first() {
                Some(Token::$i(x)) => {
                    let x = input.ctx.add_string(x);
                    Ok((input.slice(1..), x))
                }
                _ => Err(Err::Error(Error {
                    kind: ErrorKind::$e,
                    next: None,
                })),
            }
        }
    };
}

tok_parser_body!(ident, Identifier, MissingIdent);
tok_parser_body!(int, Integer, MissingInteger);
tok_parser_body!(decimal, Decimal, MissingDecimal);

macro_rules! tok_parser {
    ($n:ident, $p:pat, $e:ident) => {
        fn $n(input: Input) -> Parsed<()> {
            match input.tokens.first() {
                Some($p) => Ok((input.slice(1..), ())),
                _ => Err(Err::Error(Error {
                    kind: ErrorKind::$e,
                    next: None,
                })),
            }
        }
    };
}

tok_parser!(double_colon, Token::DoubleColon, MissingDoubleColon);
tok_parser!(comma, Token::Comma, MissingComma);
tok_parser!(left_square, Token::LeftSquare, MissingLeftSquare);
tok_parser!(right_square, Token::RightSquare, MissingRightSquare);
tok_parser!(left_curly, Token::LeftCurly, MissingLeftCurly);
tok_parser!(right_curly, Token::RightCurly, MissingRightCurly);
tok_parser!(left_round, Token::LeftRound, MissingLeftRound);
tok_parser!(right_round, Token::RightRound, MissingRightRound);
tok_parser!(arrow, Token::Arrow, MissingArrow);
tok_parser!(semicolon, Token::Semicolon, MissingSemicolon);
tok_parser!(equals, Token::Equals, MissingEquals);
tok_parser!(pipe, Token::Pipe, MissingPipe);
tok_parser!(wide_arrow, Token::WideArrow, MissingWideArrow);
tok_parser!(dot, Token::Dot, MissingDot);
tok_parser!(colon, Token::Colon, MissingColon);
tok_parser!(underscore, Token::Underscore, MissingUnderscore);
tok_parser!(tick, Token::Tick, MissingTick);
tok_parser!(left_angle, Token::LeftAngle, MissingLeftAngle);
tok_parser!(right_angle, Token::RightAngle, MissingRightAngle);

fn product_ty(input: Input) -> Parsed<Type> {
    delimited(
        left_curly,
        separated_list0(comma, int.or(ident).and(ty)),
        right_curly,
    )
    .map(Type::Product)
    .parse(input)
}

fn sum_ty(input: Input) -> Parsed<Type> {
    delimited(
        left_square,
        separated_list1(comma, int.or(ident).and(ty)),
        right_square,
    )
    .map(Type::Sum)
    .parse(input)
}

fn function_ty(input: Input) -> Parsed<Type> {
    arg.and(arrow)
        .and(ty)
        .map(|((arg, _), ret)| (arg, ret))
        .map(Type::Function)
        .parse(input)
}

fn arg(input: Input) -> Parsed<TypeId> {
    let (mut input, ty) = alt((
        delimited(left_round, function_ty, right_round),
        product_ty,
        sum_ty,
        ident.map(Type::Symbol),
    ))
    .parse(input)?;
    let ty = input.ctx.add_ty(ty);
    Ok((input, ty))
}

fn ty(input: Input) -> Parsed<TypeId> {
    let (mut input, ty) =
        alt((function_ty, product_ty, sum_ty, ident.map(Type::Symbol))).parse(input)?;
    let ty = input.ctx.add_ty(ty);
    Ok((input, ty))
}

fn tdecl(input: Input) -> Parsed<()> {
    let (mut input, (ident, ty)) = ident
        .and(double_colon)
        .and(ty)
        .and(semicolon)
        .map(|(((ident, _), ty), _)| (ident, ty))
        .parse(input)?;

    input.ctx.block_mut().symbols.insert(
        ident,
        Symbol {
            ty: Some(ty),
            value: None,
        },
    );

    Ok((input, ()))
}

fn call(input: Input) -> Parsed<Operand> {
    let (mut input, (ident, arg)) = ident
        .and(delimited(left_round, expr, right_round))
        .parse(input)?;
    let inst = input.ctx.add_inst(Inst {
        op: Opcode::Call,
        args: vec![Operand::Ident(ident), arg],
    });
    Ok((input, Operand::Value(inst)))
}

fn variable(input: Input) -> Parsed<Operand> {
    let (input, ident) = ident.parse(input)?;
    let oper = if let Some(Symbol {
        value: Some(oper), ..
    }) = input.ctx.block().symbols.get(&ident).copied()
    {
        oper
    } else {
        Operand::Ident(ident)
    };
    Ok((input, oper))
}

fn constant(input: Input) -> Parsed<Operand> {
    let (mut input, int) = int.parse(input)?;
    let inst = input.ctx.add_inst(Inst {
        op: Opcode::Int,
        args: vec![Operand::Ident(int)],
    });
    Ok((input, Operand::Value(inst)))
}

fn product(input: Input) -> Parsed<Operand> {
    let (mut input, fields) = delimited(
        left_curly,
        separated_list0(comma, int.or(ident).and(expr)),
        right_curly,
    )
    .parse(input)?;
    let inst = Inst {
        op: Opcode::Product,
        args: fields
            .into_iter()
            .map(|(ident, oper)| {
                input.ctx.add_inst(Inst {
                    op: Opcode::Field,
                    args: vec![Operand::Ident(ident), oper],
                })
            })
            .map(Operand::Value)
            .collect(),
    };
    let inst = input.ctx.add_inst(inst);
    Ok((input, Operand::Value(inst)))
}

fn variant(input: Input) -> Parsed<Operand> {
    let (mut input, (variant, oper)) = ident
        .and(delimited(left_square, expr, right_square))
        .parse(input)?;
    let inst = input.ctx.add_inst(Inst {
        op: Opcode::Variant,
        args: vec![Operand::Ident(variant), oper],
    });
    Ok((input, Operand::Value(inst)))
}

fn expr(input: Input) -> Parsed<Operand> {
    alt((lambda, product, variant, call, access, variable, constant)).parse(input)
}

fn lambda(mut input: Input) -> Parsed<Operand> {
    let parent = input.ctx.block;
    let block = input.ctx.add_block(None);
    input.ctx.block = block;

    let (mut input, (arg, _)) = ident.and(arrow).parse(input)?;
    input.ctx.block_mut().symbols.insert(
        arg,
        Symbol {
            ty: None,
            value: Some(Operand::Arg),
        },
    );

    let (mut input, value) = expr.parse(input)?;
    input.ctx.add_inst(Inst {
        op: Opcode::Return,
        args: vec![value],
    });
    input.ctx.block = parent;
    Ok((input, Operand::Block(block)))
}

fn access(input: Input) -> Parsed<Operand> {
    let (mut input, ((base, _), fields)) = variable
        .and(dot)
        .and(separated_list1(dot, int.or(ident)))
        .parse(input)?;

    let inst = fields.into_iter().fold(base, |oper, field| {
        Operand::Value(input.ctx.add_inst(Inst {
            op: Opcode::Get,
            args: vec![oper, Operand::Ident(field)],
        }))
    });
    Ok((input, inst))
}

fn vdecl(mut input: Input) -> Parsed<()> {
    let parent = input.ctx.block;
    let block = input.ctx.add_block(Some(TypeId::UNIT));
    input.ctx.block = block;

    let (mut input, (((ident, _), value), _)) =
        ident.and(equals).and(expr).and(semicolon).parse(input)?;

    input.ctx.add_inst(Inst {
        op: Opcode::Return,
        args: vec![value],
    });

    input.ctx.block = parent;
    input
        .ctx
        .block_mut()
        .symbols
        .get_mut(&ident)
        .expect("Value decl without type hint")
        .value = Some(Operand::Block(block));

    Ok((input, ()))
}

pub fn parse_tokens(tokens: &[Token]) -> Option<Context> {
    let input = Input {
        tokens,
        ctx: Context::new(),
    };
    let (input, parsed) = many0(tdecl.or(vdecl)).parse(input).unwrap();
    Some(input.ctx)
}
