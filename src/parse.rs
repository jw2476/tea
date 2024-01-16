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

#[derive(Clone)]
pub struct Input<'a> {
    ctx: Context,
    scopes: Vec<Block>,
    tokens: &'a [Token],
}

impl<'a> Input<'a> {
    pub fn slice(self, range: RangeFrom<usize>) -> Self {
        Self {
            ctx: self.ctx,
            scopes: Vec::new(),
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
tok_parser!(left_angle, Token::LeftAngle, MissingLeftAngle);
tok_parser!(right_angle, Token::RightAngle, MissingRightAngle);

fn path(input: Input) -> Parsed<Vec<String>> {
    separated_list1(double_colon, ident).parse(input)
}

pub type ExprId = usize;
pub type TypeId = usize;
#[derive(Clone, Debug)]
pub struct Sum<T>(pub Vec<(String, T)>);
#[derive(Clone, Debug)]
pub struct Product<T>(pub Vec<(String, T)>);
pub type Call = (String, ExprId);
pub type Function = (TypeId, TypeId);
pub type Lambda = (ExprId, ExprId);
pub type Match = (ExprId, Vec<ExprId>);
pub type Access = (ExprId, String);
pub type Variant = (String, ExprId);
pub type Block = (Vec<Decl>, Option<ExprId>);

#[derive(Clone, Debug)]
pub enum Expr {
    Arg,
    Sum(Sum<ExprId>),
    Product(Product<ExprId>),
    Call(Call),
    Lambda(Lambda),
    Match(Match),
    Access(Access),
    Unreachable,
    Ignore,
    Var((ExprId, TypeId)),
    Symbol((String, Vec<String>)),
    Int(String),
    Decimal(String),
    TypeVar(Option<String>),
    Variant(Variant),
    Block(Block),
}

#[derive(Clone, Debug)]
pub enum Type {
    Sum(Sum<TypeId>),
    Product(Product<TypeId>),
    Function(Function),
    Named((String, Vec<String>)),
    Var(String),
}

#[derive(Clone, Debug)]
pub struct Decl {
    pub ident: String,
    pub generics: Vec<String>,
    pub ty: TypeId,
    pub value: Option<ExprId>,
}

#[derive(Clone, Debug)]
pub struct Context {
    pub exprs: Vec<Expr>,
    pub types: Vec<Type>,
}

impl Context {
    pub fn add_expr(&mut self, expr: Expr) -> ExprId {
        self.exprs.push(expr);
        self.exprs.len() - 1
    }

    pub fn add_ty(&mut self, ty: Type) -> ExprId {
        self.types.push(ty);
        self.types.len() - 1
    }
}

fn variable(input: Input) -> Parsed<ExprId> {
    let (mut input, (ident, generics)) = ident.and(generics).parse(input)?;
    let expr = if let Some(decl) = input
        .scopes
        .last()
        .unwrap()
        .0
        .iter()
        .find(|d| d.ident == ident)
    {
        Expr::Var((decl.value.expect("referenced decl with no value"), decl.ty))
    } else {
        Expr::Symbol((ident, generics))
    };
    let expr = input.ctx.add_expr(expr);
    Ok((input, expr))
}

fn sum_expr(input: Input) -> Parsed<Sum<ExprId>> {
    delimited(
        left_square,
        separated_list1(comma, ident.and(expr)),
        right_square,
    )
    .map(Sum)
    .parse(input)
}

fn opt_ty(input: Input) -> Parsed<TypeId> {
    let (mut input, ty) = opt(ty).parse(input)?;
    if let Some(ty) = ty {
        return Ok((input, ty));
    }
    let ty = input.ctx.add_ty(Type::Product(Product(Vec::new())));
    Ok((input, ty))
}

fn sum_ty(input: Input) -> Parsed<Sum<TypeId>> {
    delimited(
        left_square,
        separated_list1(comma, ident.and(opt_ty)),
        right_square,
    )
    .map(Sum)
    .parse(input)
}

fn product_expr(input: Input) -> Parsed<Product<ExprId>> {
    let labelled = delimited(
        left_curly,
        separated_list0(comma, ident.or(int).and(expr)),
        right_curly,
    )
    .map(Product);

    let unlabelled =
        delimited(left_round, separated_list0(comma, expr), right_round).map(|exprs| {
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

fn product_ty(input: Input) -> Parsed<Product<TypeId>> {
    let labelled = delimited(
        left_curly,
        separated_list0(comma, ident.or(int).and(ty)),
        right_curly,
    )
    .map(Product);

    let unlabelled = delimited(left_round, separated_list0(comma, ty), right_round).map(|exprs| {
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

    let product = input.ctx.add_expr(Expr::Product(Product(
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
        .and(ty)
        .map(|((x, _), y)| (x, y))
        .parse(input)
}

fn lambda(input: Input) -> Parsed<Lambda> {
    pattern
        .and(arrow)
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
            separated_list1(comma, lambda),
            right_curly,
        ))
        .map(|((_, x), y)| (x, y))
        .parse(input)?;
    let branches = branches
        .into_iter()
        .map(|branch| input.ctx.add_expr(Expr::Lambda(branch)))
        .collect();
    Ok((input, (expr, branches)))
}

fn variant(input: Input) -> Parsed<Variant> {
    fn opt_expr(input: Input) -> Parsed<ExprId> {
        let (mut input, expr) = opt(expr).parse(input)?;
        let expr = match expr {
            Some(expr) => expr,
            None => input.ctx.add_expr(Expr::Product(Product(Vec::new()))),
        };
        Ok((input, expr))
    }

    ident
        .and(delimited(left_square, opt_expr, right_square))
        .parse(input)
}

fn block(mut input: Input) -> Parsed<Block> {
    input.scopes.push((Vec::new(), None));
    let (mut input, (_, expr)) =
        delimited(left_curly, many0(decl).and(opt(expr)), right_curly).parse(input)?;
    input.scopes.last_mut().unwrap().1 = expr;
    let block = input.scopes.pop().unwrap();
    Ok((input, block))
}

fn ty(input: Input) -> Parsed<TypeId> {
    let (mut input, ty) = alt((
        function.map(Type::Function),
        sum_ty.map(Type::Sum),
        product_ty.map(Type::Product),
        ident.and(generics).map(Type::Named),
        ident.map(Type::Var),
    ))
    .parse(input)?;
    let id = input.ctx.add_ty(ty);
    Ok((input, id))
}

fn arg(input: Input) -> Parsed<TypeId> {
    let (mut input, ty) = alt((
        sum_ty.map(Type::Sum),
        product_ty.map(Type::Product),
        delimited(left_round, function, right_round).map(Type::Function),
        ident.and(generics).map(Type::Named),
        ident.map(Type::Var),
    ))
    .parse(input)?;
    let id = input.ctx.add_ty(ty);
    Ok((input, id))
}

fn base(input: Input) -> Parsed<ExprId> {
    let inner = |input| {
        let (mut input, expr) = alt((
            sum_expr.map(Expr::Sum),
            product_expr.map(Expr::Product),
            call.map(Expr::Call),
            variant.map(Expr::Variant),
            pmatch.map(Expr::Match),
        ))
        .parse(input)?;
        let id = input.ctx.add_expr(expr);
        Ok((input, id))
    };
    inner.or(variable).parse(input)
}

enum Part {
    Call(Call),
    Access(String),
}

fn complex_expr(input: Input) -> Parsed<ExprId> {
    let (mut input, (base, parts)) = base
        .and(dot)
        .and(separated_list1(
            dot,
            call.map(Part::Call).or(int.or(ident).map(Part::Access)),
        ))
        .map(|((expr, _), parts)| (expr, parts))
        .parse(input)?;

    let id = parts.into_iter().fold(base, |expr, part| match part {
        Part::Call((func, arg)) => {
            let arg = if let Expr::Product(Product(fields)) = &mut input.ctx.exprs[arg] {
                *fields = [expr]
                    .iter()
                    .chain(fields.iter().map(|(_, x)| x))
                    .enumerate()
                    .map(|(i, field)| (i.to_string(), *field))
                    .collect();
                arg
            } else {
                input.ctx.add_expr(Expr::Product(Product(vec![
                    ("0".to_string(), expr),
                    ("1".to_string(), arg),
                ])))
            };
            input.ctx.add_expr(Expr::Call((func, arg)))
        }
        Part::Access(field) => input.ctx.add_expr(Expr::Access((expr, field))),
    });
    Ok((input, id))
}

fn pattern(input: Input) -> Parsed<ExprId> {
    let (mut input, expr) = alt((
        sum_expr.map(Expr::Sum),
        product_expr.map(Expr::Product),
        variant.map(Expr::Variant),
        ident.map(|x| Expr::Symbol((x, Vec::new()))),
        underscore.map(|_| Expr::Ignore),
    ))
    .parse(input)?;
    let id = input.ctx.add_expr(expr);
    Ok((input, id))
}

fn _expr(input: Input) -> Parsed<ExprId> {
    let inner = |input| {
        let (mut input, expr) = alt((
            lambda.map(Expr::Lambda),
            sum_expr.map(Expr::Sum),
            block.map(Expr::Block),
            product_expr.map(Expr::Product),
            call.map(Expr::Call),
            variant.map(Expr::Variant),
            pmatch.map(Expr::Match),
            keyword("unreachable").map(|_| Expr::Unreachable),
            int.map(Expr::Int),
            decimal.map(Expr::Decimal),
        ))
        .parse(input)?;
        let id = input.ctx.add_expr(expr);
        Ok((input, id))
    };
    inner.or(variable).parse(input)
}

fn expr(input: Input) -> Parsed<ExprId> {
    complex_expr.or(_expr).parse(input)
}

fn fdecl(input: Input) -> Parsed<Decl> {
    let Token::Identifier(x) = input.tokens[0].clone() else {
        return Err(Err::Error(Error {
            kind: ErrorKind::MissingIdent,
            next: None,
        }));
    };
    let (mut input, ((ident, generics, ty), branches)) = tdecl
        .and(many1(
            keyword(&x)
                .and(lambda)
                .and(semicolon)
                .map(|((_, function), _)| function),
        ))
        .parse(input)?;
    let Type::Function((arg, ret)) = input.ctx.types[ty] else {
        return Err(Err::Error(Error {
            kind: ErrorKind::MissingArrow,
            next: None,
        }));
    };

    let arg_expr = input.ctx.add_expr(Expr::Arg);
    input.scopes.last_mut().unwrap().0.push(Decl {
        ident: "x".to_string(),
        generics: Vec::new(),
        ty: arg,
        value: Some(arg_expr),
    });

    let x = input.ctx.add_expr(Expr::Var((arg_expr, arg)));
    let branches = branches
        .into_iter()
        .map(|branch| input.ctx.add_expr(Expr::Lambda(branch)))
        .collect();
    let m = input.ctx.add_expr(Expr::Match((x, branches)));
    let function = input.ctx.add_expr(Expr::Lambda((arg_expr, m)));
    Ok((
        input,
        Decl {
            ident,
            generics,
            ty,
            value: Some(function),
        },
    ))
}

fn generics(input: Input) -> Parsed<Vec<String>> {
    opt(delimited(
        left_angle,
        separated_list1(comma, ident),
        right_angle,
    ))
    .map(|x| x.unwrap_or_default())
    .parse(input)
}

fn tdecl(input: Input) -> Parsed<(String, Vec<String>, TypeId)> {
    ident
        .and(generics)
        .and(double_colon)
        .and(ty)
        .and(semicolon)
        .map(|((((ident, generics), _), ty), _)| (ident, generics, ty))
        .parse(input)
}

fn decl(input: Input) -> Parsed<()> {
    let combined = ident
        .and(generics)
        .and(colon)
        .and(ty)
        .and(equals)
        .and(expr)
        .and(semicolon)
        .map(|((((((ident, generics), _), ty), _), value), _)| Decl {
            ident,
            generics,
            ty,
            value: Some(value),
        });

    let vdecl = ident
        .and(equals)
        .and(expr)
        .and(semicolon)
        .map(|(((ident, _), value), _)| (ident, value));

    let split = tdecl.and(opt(vdecl)).map(|(tdecl, vdecl)| Decl {
        ident: tdecl.0,
        generics: tdecl.1,
        ty: tdecl.2,
        value: vdecl.map(|x| x.1),
    });

    let (mut input, decl) = combined.or(split).or(fdecl).parse(input)?;
    input.scopes.last_mut().unwrap().0.push(decl);
    Ok((input, ()))
}

pub fn parse_tokens(tokens: &[Token]) -> Option<(Context, Vec<Decl>)> {
    let (input, block) = block
        .parse(Input {
            ctx: Context {
                exprs: Vec::new(),
                types: Vec::new(),
            },
            scopes: Vec::new(),
            tokens,
        })
        .unwrap();
    //.ok()?;
    Some((input.ctx, block.0))
}
