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

#[derive(Clone, Debug)]
pub struct NodeId<T> {
    id: usize,
    phantom: PhantomData<T>,
}
pub type UntypedNodeId = usize;
impl<T> From<NodeId<T>> for UntypedNodeId {
    fn from(value: NodeId<T>) -> Self {
        value.id
    }
}

#[derive(Clone, Debug)]
pub struct Sum(Vec<(String, NodeId<Expr>)>);
#[derive(Clone, Debug)]
pub struct Product(Vec<(String, NodeId<Expr>)>);
pub type Call = (String, NodeId<Expr>);
pub type Function = (NodeId<Expr>, NodeId<Expr>);
pub type Match = (NodeId<Expr>, Vec<NodeId<Function>>);
pub type Access = (NodeId<Expr>, String);

#[derive(Clone, Debug)]
pub enum Expr {
    Sum(NodeId<Sum>),
    Product(NodeId<Product>),
    Call(NodeId<Call>),
    Function(NodeId<Function>),
    Match(NodeId<Match>),
    Access(NodeId<Access>),
    Type,
    Unreachable,
    Var(Option<String>),
    Int(String),
    Decimal(String),
    TypeVar(Option<String>),
}

#[derive(Clone, Debug)]
pub struct Decl {
    ident: String,
    ty: NodeId<Expr>,
    value: NodeId<Expr>,
}

#[derive(Clone, Debug)]
pub enum Node {
    Sum(Sum),
    Product(Product),
    Call(Call),
    Function(Function),
    Match(Match),
    Expr(Expr),
    Access(Access),
}

#[derive(Clone, Debug)]
pub struct Context {
    nodes: Vec<Node>,
}

impl Context {
    pub fn add_node<T>(&mut self, node: T) -> NodeId<T>
    where
        T: Into<Node>,
    {
        self.nodes.push(node.into());
        NodeId {
            id: self.nodes.len() - 1,
            phantom: PhantomData,
        }
    }
}

trait NodeIndex<T> {
    fn get(&self, id: NodeId<T>) -> Option<&T>;
}

macro_rules! impl_node {
    ($t:ident) => {
        impl NodeIndex<$t> for Context {
            fn get(&self, id: NodeId<$t>) -> Option<&$t> {
                match &self.nodes[id.id] {
                    Node::$t(x) => Some(x),
                    _ => None,
                }
            }
        }

        impl From<$t> for Node {
            fn from(value: $t) -> Self {
                Node::$t(value)
            }
        }
    };
}

impl_node!(Sum);
impl_node!(Product);
impl_node!(Call);
impl_node!(Function);
impl_node!(Match);
impl_node!(Expr);
impl_node!(Access);

fn sum(input: Input) -> Parsed<NodeId<Sum>> {
    let (mut input, sum) = delimited(
        left_square,
        separated_list1(comma, ident.and(expr)),
        right_square,
    )
    .map(Sum)
    .parse(input)?;
    let id = input.ctx.add_node(sum);
    Ok((input, id))
}

fn product(input: Input) -> Parsed<NodeId<Product>> {
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

    let (mut input, product) = labelled.or(unlabelled).parse(input)?;
    let id = input.ctx.add_node(product);
    Ok((input, id))
}

fn _call_single(input: Input) -> Parsed<NodeId<Call>> {
    let (mut input, call) = ident
        .and(delimited(left_round, expr, right_round))
        .parse(input)?;
    let id = input.ctx.add_node(call);
    Ok((input, id))
}

fn _call_multiple(input: Input) -> Parsed<NodeId<Call>> {
    let (mut input, (arg, exprs)) = ident
        .and(delimited(
            left_round,
            separated_list0(comma, expr),
            right_round,
        ))
        .parse(input)?;

    let product = input.ctx.add_node(Product(
        exprs
            .into_iter()
            .enumerate()
            .map(|(i, expr)| (i.to_string(), expr))
            .collect(),
    ));
    let expr = input.ctx.add_node(Expr::Product(product));
    let id = input.ctx.add_node((arg, expr));
    Ok((input, id))
}

fn call(input: Input) -> Parsed<NodeId<Call>> {
    _call_single.or(_call_multiple).parse(input)
}

fn function(input: Input) -> Parsed<NodeId<Function>> {
    let (mut input, function) = arg
        .and(arrow)
        .and(expr)
        .map(|((x, _), y)| (x, y))
        .parse(input)?;
    let id = input.ctx.add_node(function);
    Ok((input, id))
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

fn pmatch(input: Input) -> Parsed<NodeId<Match>> {
    let (mut input, m) = keyword("match")
        .and(expr)
        .and(delimited(
            left_curly,
            separated_list1(comma, function),
            right_curly,
        ))
        .map(|((_, x), y)| (x, y))
        .parse(input)?;
    let id = input.ctx.add_node(m);
    Ok((input, id))
}

fn access(input: Input) -> Parsed<NodeId<Access>> {
    let (mut input, access) = base
        .and(dot)
        .and(ident.or(int))
        .map(|((x, _), y)| (x, y))
        .parse(input)?;
    let id = input.ctx.add_node(access);
    Ok((input, id))
}

fn arg(input: Input) -> Parsed<NodeId<Expr>> {
    let (mut input, expr) = alt((
        sum.map(Expr::Sum),
        product.map(Expr::Product),
        call.map(Expr::Call),
        delimited(left_round, function, right_round).map(Expr::Function),
        pmatch.map(Expr::Match),
        access.map(Expr::Access),
        keyword("type").map(|_| Expr::Type),
        keyword("unreachable").map(|_| Expr::Unreachable),
        ident.map(|x| Expr::Var(Some(x))),
        underscore.map(|_| Expr::Var(None)),
        int.map(Expr::Int),
        decimal.map(Expr::Decimal),
        tick.and(ident).map(|(_, x)| Expr::TypeVar(Some(x))),
        tick.and(underscore).map(|_| Expr::TypeVar(None)),
    ))
    .parse(input)?;
    let id = input.ctx.add_node(expr);
    Ok((input, id))
}

fn base(input: Input) -> Parsed<NodeId<Expr>> {
    let (mut input, expr) = alt((
        sum.map(Expr::Sum),
        product.map(Expr::Product),
        call.map(Expr::Call),
        pmatch.map(Expr::Match),
        ident.map(|x| Expr::Var(Some(x))),
        tick.and(ident).map(|(_, x)| Expr::TypeVar(Some(x))),
        tick.and(underscore).map(|_| Expr::TypeVar(None)),
    ))
    .parse(input)?;
    let id = input.ctx.add_node(expr);
    Ok((input, id))
}

fn expr(input: Input) -> Parsed<NodeId<Expr>> {
    let (mut input, expr) = alt((
        function.map(Expr::Function),
        sum.map(Expr::Sum),
        product.map(Expr::Product),
        call.map(Expr::Call),
        pmatch.map(Expr::Match),
        access.map(Expr::Access),
        keyword("type").map(|_| Expr::Type),
        keyword("unreachable").map(|_| Expr::Unreachable),
        ident.map(|x| Expr::Var(Some(x))),
        underscore.map(|_| Expr::Var(None)),
        int.map(Expr::Int),
        decimal.map(Expr::Decimal),
        tick.and(ident).map(|(_, x)| Expr::TypeVar(Some(x))),
        tick.and(underscore).map(|_| Expr::TypeVar(None)),
    ))
    .parse(input)?;
    let id = input.ctx.add_node(expr);
    Ok((input, id))
}

fn decl(input: Input) -> Parsed<Decl> {
    ident
        .and(colon)
        .and(expr)
        .and(equals)
        .and(expr)
        .and(semicolon)
        .map(|(((((ident, _), ty), _), value), _)| Decl { ident, ty, value })
        .parse(input)
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
