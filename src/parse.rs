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

fn ident(input: Input) -> Parsed<String> {
    match input.tokens.first() {
        Some(Token::Identifier(ident)) => Ok((input.slice(1..), ident.clone())),
        _ => Err(Err::Error(Error {
            kind: ErrorKind::MissingIdent,
            next: None,
        })),
    }
}

fn int(input: Input) -> Parsed<String> {
    match input.tokens.first() {
        Some(Token::Integer(int)) => Ok((input.slice(1..), int.clone())),
        _ => Err(Err::Error(Error {
            kind: ErrorKind::MissingInteger,
            next: None,
        })),
    }
}

fn decimal(input: Input) -> Parsed<String> {
    match input.tokens.first() {
        Some(Token::Decimal(int)) => Ok((input.slice(1..), int.clone())),
        _ => Err(Err::Error(Error {
            kind: ErrorKind::MissingDecimal,
            next: None,
        })),
    }
}

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Path {
    pub parts: Vec<String>,
}

fn path(input: Input) -> Parsed<Path> {
    separated_list1(double_colon, ident)
        .map(|parts| Path { parts })
        .parse(input)
}

pub type NodeId = usize;
#[derive(Clone, Debug)]
pub struct Sum(pub Vec<(String, NodeId)>);
#[derive(Clone, Debug)]
pub struct Product(pub Vec<(String, NodeId)>);
pub type Call = (String, NodeId);
pub type Function = (NodeId, NodeId);
pub type Match = (NodeId, Vec<NodeId>);
pub type Access = (NodeId, String);

#[derive(Clone, Debug)]
pub enum Node {
    Sum(Sum),
    Product(Product),
    Call(Call),
    Function(Function),
    Match(Match),
    Access(Access),
    Type,
    Unreachable,
    Var(Option<String>),
    Int(String),
    Decimal(String),
    TypeVar(Option<String>),
}

#[derive(Clone, Debug)]
pub struct Decl {
    pub ident: String,
    pub ty: NodeId,
    pub value: NodeId,
}

#[derive(Clone, Debug)]
pub struct Context {
    pub nodes: Vec<Node>,
}

impl Context {
    pub fn add_node(&mut self, node: Node) -> NodeId
where {
        self.nodes.push(node.into());
        self.nodes.len() - 1
    }
    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id)
    }
}

fn sum(input: Input) -> Parsed<Sum> {
    delimited(
        left_square,
        separated_list1(comma, ident.and(expr)),
        right_square,
    )
    .map(Sum)
    .parse(input)
}

fn product(input: Input) -> Parsed<Product> {
    let labelled = delimited(
        left_curly,
        separated_list0(comma, ident.or(int).and(expr)),
        right_curly,
    )
    .map(Product);

    let unlabelled =
        delimited(left_curly, separated_list0(comma, expr), right_curly).map(|exprs| {
            Product(
                exprs
                    .into_iter()
                    .enumerate()
                    .map(|(i, expr)| (i.to_string(), expr))
                    .collect(),
            )
        });

    labelled.or(unlabelled).parse(input)
}

fn _call_single(input: Input) -> Parsed<Call> {
    ident
        .and(delimited(left_round, expr, right_round))
        .parse(input)
}

fn _call_multiple(input: Input) -> Parsed<Call> {
    let (mut input, (arg, exprs)) = ident
        .and(delimited(
            left_round,
            separated_list0(comma, expr),
            right_round,
        ))
        .parse(input)?;

    let product = input.ctx.add_node(Node::Product(Product(
        exprs
            .into_iter()
            .enumerate()
            .map(|(i, expr)| (i.to_string(), expr))
            .collect(),
    )));
    Ok((input, (arg, product)))
}

fn call(input: Input) -> Parsed<Call> {
    _call_single.or(_call_multiple).parse(input)
}

fn function(input: Input) -> Parsed<Function> {
    arg.and(arrow)
        .and(expr)
        .map(|((x, _), y)| (x, y))
        .parse(input)
}

fn keyword<T: ToString>(word: T) -> impl Fn(Input) -> Parsed<()> {
    move |input| match input.tokens.first() {
        Some(Token::Identifier(w)) if w == &word.to_string() => Ok((input.slice(1..), ())),
        _ => Err(Err::Error(Error {
            kind: ErrorKind::MissingKeyword(word.to_string()),
            next: None,
        })),
    }
}

fn pmatch(input: Input) -> Parsed<Match> {
    let (mut input, (expr, branches)) = keyword("match")
        .and(expr)
        .and(delimited(
            left_curly,
            separated_list1(comma, function),
            right_curly,
        ))
        .map(|((_, x), y)| (x, y))
        .parse(input)?;
    let branches = branches
        .into_iter()
        .map(|branch| input.ctx.add_node(Node::Function(branch)))
        .collect();
    Ok((input, (expr, branches)))
}

fn access(input: Input) -> Parsed<Access> {
    base.and(dot)
        .and(ident.or(int))
        .map(|((x, _), y)| (x, y))
        .parse(input)
}

fn arg(input: Input) -> Parsed<NodeId> {
    let (mut input, expr) = alt((
        sum.map(Node::Sum),
        product.map(Node::Product),
        call.map(Node::Call),
        delimited(left_round, function, right_round).map(Node::Function),
        pmatch.map(Node::Match),
        access.map(Node::Access),
        keyword("type").map(|_| Node::Type),
        keyword("unreachable").map(|_| Node::Unreachable),
        ident.map(|x| Node::Var(Some(x))),
        underscore.map(|_| Node::Var(None)),
        int.map(Node::Int),
        decimal.map(Node::Decimal),
        tick.and(ident).map(|(_, x)| Node::TypeVar(Some(x))),
        tick.and(underscore).map(|_| Node::TypeVar(None)),
    ))
    .parse(input)?;
    let id = input.ctx.add_node(expr);
    Ok((input, id))
}

fn base(input: Input) -> Parsed<NodeId> {
    let (mut input, expr) = alt((
        sum.map(Node::Sum),
        product.map(Node::Product),
        call.map(Node::Call),
        pmatch.map(Node::Match),
        ident.map(|x| Node::Var(Some(x))),
        tick.and(ident).map(|(_, x)| Node::TypeVar(Some(x))),
        tick.and(underscore).map(|_| Node::TypeVar(None)),
    ))
    .parse(input)?;
    let id = input.ctx.add_node(expr);
    Ok((input, id))
}

fn expr(input: Input) -> Parsed<NodeId> {
    let (mut input, expr) = alt((
        function.map(Node::Function),
        sum.map(Node::Sum),
        product.map(Node::Product),
        call.map(Node::Call),
        pmatch.map(Node::Match),
        access.map(Node::Access),
        keyword("type").map(|_| Node::Type),
        keyword("unreachable").map(|_| Node::Unreachable),
        ident.map(|x| Node::Var(Some(x))),
        underscore.map(|_| Node::Var(None)),
        int.map(Node::Int),
        decimal.map(Node::Decimal),
        tick.and(ident).map(|(_, x)| Node::TypeVar(Some(x))),
        tick.and(underscore).map(|_| Node::TypeVar(None)),
    ))
    .parse(input)?;
    let id = input.ctx.add_node(expr);
    Ok((input, id))
}

fn fdecl(input: Input) -> Parsed<(String, NodeId)> {
    let Token::Identifier(x) = input.tokens[0].clone() else {
        return Err(Err::Error(Error {
            kind: ErrorKind::MissingIdent,
            next: None,
        }));
    };
    let (mut input, branches) = many1(
        keyword(&x)
            .and(function)
            .and(semicolon)
            .map(|((_, function), _)| function),
    )
    .parse(input)?;
    let x1 = input.ctx.add_node(Node::Var(Some("x".to_string())));
    let x2 = input.ctx.add_node(Node::Var(Some("x".to_string())));
    let branches = branches
        .into_iter()
        .map(|branch| input.ctx.add_node(Node::Function(branch)))
        .collect();
    let m = input.ctx.add_node(Node::Match((x1, branches)));
    let function = input.ctx.add_node(Node::Function((x2, m)));
    Ok((input, (x, function)))
}

fn decl(input: Input) -> Parsed<Decl> {
    let combined = ident
        .and(colon)
        .and(expr)
        .and(equals)
        .and(expr)
        .and(semicolon)
        .map(|(((((ident, _), ty), _), value), _)| Decl { ident, ty, value });

    let tdecl = ident
        .and(double_colon)
        .and(expr)
        .and(semicolon)
        .map(|(((ident, _), ty), _)| (ident, ty));
    let vdecl = ident
        .and(equals)
        .and(expr)
        .and(semicolon)
        .map(|(((ident, _), value), _)| (ident, value));

    let split = tdecl.and(vdecl.or(fdecl)).map(|(tdecl, vdecl)| Decl {
        ident: tdecl.0,
        ty: tdecl.1,
        value: vdecl.1,
    });

    combined.or(split).parse(input)
}

pub fn parse_tokens(tokens: &[Token]) -> Option<(Context, Vec<Decl>)> {
    let (input, decls) = many0(decl)
        .parse(Input {
            ctx: Context { nodes: Vec::new() },
            tokens,
        })
        .ok()?;
    Some((input.ctx, decls))
}
