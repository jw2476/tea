use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::{Debug, Display},
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

pub type TypeId = usize;
pub type ScopeId = usize;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
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

    Product(Product<TypeId>),
    Sum(Sum<TypeId>),
    Function(TypeId, TypeId),
    Named(Path, Option<TypeId>), // e.g. std::io::Writer, the option stores the resolved type
    Infer,
}

#[derive(Debug)]
pub enum ExprKind {
    Block(ScopeId),
    Product(Product<Expr>),
    Variant(Variant<Expr>),
    Lambda(Lambda),
    Call(Call),
    Match(Match),
    Variable(String),
    Int(String),
    Decimal(String),
    Access(Box<Expr>, String),
}

impl ExprKind {
    pub fn infer(self, input: &Input) -> Expr {
        Expr {
            kind: self,
            ty: input.ctx.borrow_mut().add_infer(),
        }
    }
}

#[derive(Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub ty: TypeId,
}

#[derive(Debug)]
pub enum ScopeKind {
    Module,
    Block(Expr),
    Arm(Pattern, Expr),
    Incomplete,
}

#[derive(Debug)]
pub struct Scope {
    pub kind: ScopeKind,
    pub types: HashMap<String, TypeId>,
    pub values: HashMap<String, Expr>,
    pub parent: Option<ScopeId>,
}

impl Scope {
    pub fn find_type(&self, ctx: &Context, ident: &str) -> Option<TypeId> {
        match self.types.get(ident) {
            Some(ty) => Some(*ty),
            None => match self.parent {
                Some(p) => ctx.scopes[p].find_type(ctx, ident),
                None => None,
            },
        }
    }
}

#[derive(Default, Debug)]
pub struct Context {
    pub types: Vec<Type>,
    pub scopes: Vec<Scope>,
}

impl Context {
    /*
    pub fn add_type(&mut self, ty: crate::parse::Type) -> TypeId {
        let ty = match ty {
            crate::parse::Type::Sum(Sum { variants }) => Type::Sum(Sum {
                variants: variants
                    .into_iter()
                    .map(|(l, t)| (l, self.add_type(t)))
                    .collect(),
            }),
            crate::parse::Type::Product(Product { fields }) => Type::Product(Product {
                fields: fields
                    .into_iter()
                    .map(|(l, t)| (l, self.add_type(t)))
                    .collect(),
            }),
            crate::parse::Type::Function(Function {
                argument,
                return_type,
            }) => Type::Function(self.add_type(*argument), self.add_type(*return_type)),
            crate::parse::Type::Path(Path { parts }) => match parts[0].as_str() {
                "u8" => Type::U8,
                "u16" => Type::U16,
                "u32" => Type::U32,
                "u64" => Type::U64,
                "usize" => Type::Usize,
                "i8" => Type::I8,
                "i16" => Type::I16,
                "i32" => Type::I32,
                "i64" => Type::I64,
                "isize" => Type::Isize,
                "f32" => Type::F32,
                "f64" => Type::F64,
                _ => Type::Named(Path { parts }, None),
            },
        };

        if let Some(id) = self.types.iter().position(|&x| x == ty) {
            id
        } else {
            self.types.push(ty);
            self.types.len() - 1
        }
    }
    */

    pub fn add_type(&mut self, ty: Type) -> TypeId {
        if let Some(id) = self.types.iter().position(|x| x == &ty) {
            id
        } else {
            self.types.push(ty);
            self.types.len() - 1
        }
    }

    pub fn add_infer(&mut self) -> TypeId {
        self.types.push(Type::Infer);
        self.types.len() - 1
    }

    pub fn add_unit(&mut self) -> TypeId {
        let ty = Type::Product(Product { fields: Vec::new() });
        self.add_type(ty)
    }

    pub fn add_scope(&mut self, scope: Scope) -> ScopeId {
        self.scopes.push(scope);
        self.scopes.len() - 1
    }

    pub fn get_unit(&self) -> Option<TypeId> {
        let ty = Type::Product(Product { fields: Vec::new() });
        self.types.iter().position(|t| t == &ty)
    }
}

#[derive(Clone)]
pub struct Input<'a> {
    tokens: &'a [Token],
}

impl<'a> Input<'a> {
    pub fn slice(self, range: RangeFrom<usize>) -> Self {
        Self {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Path {
    pub parts: Vec<String>,
}

fn path(input: Input) -> Parsed<Path> {
    separated_list1(double_colon, ident)
        .map(|parts| Path { parts })
        .parse(input)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sum<T> {
    pub variants: Vec<(String, T)>,
}

fn sum(input: Input) -> Parsed<Sum<TypeId>> {
    let field = opt(ty).map(|x| x.unwrap_or_else(|| input.ctx.borrow_mut().add_unit()));

    delimited(
        left_square,
        separated_list1(comma, ident.and(field)),
        right_square,
    )
    .map(|variants| Sum { variants })
    .parse(input.clone())
}

struct ProductParser<F> {
    f: F,
}

impl<'a, F, T> Parser<Input<'a>, Product<T>, Error> for ProductParser<F>
where
    F: Parser<Input<'a>, T, Error> + Copy,
{
    fn parse(&mut self, input: Input<'a>) -> Parsed<'a, Product<T>> {
        let labelled = delimited(
            left_curly,
            separated_list0(comma, (ident.or(int)).and(self.f)),
            right_curly,
        )
        .map(|fields| Product { fields });

        let unlabelled =
            delimited(left_curly, separated_list0(comma, self.f), right_curly).map(|fields| {
                Product {
                    fields: fields
                        .into_iter()
                        .enumerate()
                        .map(|(i, field)| (i.to_string(), field))
                        .collect(),
                }
            });

        labelled.or(unlabelled).parse(input)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Product<T> {
    pub fields: Vec<(String, T)>,
}

fn product<'a, T, F>(f: F) -> ProductParser<F>
where
    F: Parser<Input<'a>, T, Error>,
{
    ProductParser { f }
}

fn function(input: Input) -> Parsed<(TypeId, TypeId)> {
    argument
        .and(arrow)
        .and(ty)
        .map(|((x, _), y)| (x, y))
        .parse(input)
}

fn argument(input: Input) -> Parsed<TypeId> {
    alt((
        sum.map(Type::Sum),
        product(ty).map(Type::Product),
        path.map(|p| Type::Named(p, None)),
        delimited(left_round, function, right_round).map(|(x, y)| Type::Function(x, y)),
    ))
    .map(|ty| input.ctx.borrow_mut().add_type(ty))
    .parse(input.clone())
}

fn ty(input: Input) -> Parsed<TypeId> {
    alt((
        function.map(|(x, y)| Type::Function(x, y)),
        sum.map(Type::Sum),
        product(ty).map(Type::Product),
        path.map(|p| Type::Named(p, None)),
    ))
    .map(|ty| input.ctx.borrow_mut().add_type(ty))
    .parse(input.clone())
}

#[derive(Debug)]
pub struct TypeDecl {
    pub ident: String,
    pub ty: TypeId,
}

fn tdecl(input: Input) -> Parsed<()> {
    let (input, ty) = ident
        .and(double_colon)
        .and(ty)
        .and(semicolon)
        .map(|(((ident, _), ty), _)| TypeDecl { ident, ty })
        .parse(input)?;
    input.ctx.borrow_mut().scopes[input.scope]
        .types
        .insert(ty.ident, ty.ty);
    Ok((input, ()))
}

#[derive(Debug)]
pub struct Lambda {
    pub param: String,
    pub body: Box<Expr>,
}

fn lambda(input: Input) -> Parsed<Lambda> {
    delimited(pipe, ident, pipe)
        .and(expr)
        .map(|(param, body)| Lambda {
            param,
            body: Box::new(body),
        })
        .parse(input)
}

#[derive(Debug)]
pub struct Call {
    pub func: String,
    pub arg: Box<Expr>,
}

fn call(input: Input) -> Parsed<Call> {
    // add({x, 3})
    let single = ident
        .and(delimited(left_round, expr, right_round))
        .map(|(func, arg)| Call {
            func,
            arg: Box::new(arg),
        });

    // add(x, 3) => add({x, 3}) or finish() => finish({})
    let multiple = ident
        .and(delimited(
            left_round,
            separated_list0(comma, expr),
            right_round,
        ))
        .map(|(func, args)| Call {
            func,
            arg: Box::new(
                ExprKind::Product(Product {
                    fields: args
                        .into_iter()
                        .enumerate()
                        .map(|(i, arg)| (i.to_string(), arg))
                        .collect(),
                })
                .infer(&input),
            ),
        });

    single.or(multiple).parse(input.clone())
}

struct VariantParser<F> {
    f: F,
}

impl<'a, F, T> Parser<Input<'a>, Variant<T>, Error> for VariantParser<F>
where
    F: Parser<Input<'a>, T, Error> + Copy,
{
    fn parse(&mut self, input: Input<'a>) -> IResult<Input<'a>, Variant<T>, Error> {
        path.and(self.f)
            .map(|(path, value)| Variant {
                path,
                value: Box::new(value),
            })
            .parse(input)
    }
}

#[derive(Debug)]
pub struct Variant<T> {
    pub path: Path,
    pub value: Box<T>,
}

fn variant<'a, T, F>(f: F) -> VariantParser<F>
where
    F: Parser<Input<'a>, T, Error>,
{
    VariantParser { f }
}

#[derive(Debug)]
pub enum Pattern {
    Product(Product<Self>),
    Variant(Variant<Self>),
    Variable(String),
    Int(String),
    Decimal(String),
}
fn pattern(input: Input) -> Parsed<Pattern> {
    alt((
        product(pattern).map(Pattern::Product),
        variant(pattern).map(Pattern::Variant),
        ident.map(Pattern::Variable),
        int.map(Pattern::Int),
        decimal.map(Pattern::Decimal),
    ))(input)
}

#[derive(Debug)]
pub struct Match {
    pub expr: Box<Expr>,
    pub branches: Vec<ScopeId>,
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
    keyword("match")
        .and(expr)
        .and(delimited(
            left_curly,
            separated_list1(
                comma,
                pattern
                    .and(wide_arrow)
                    .and(expr)
                    .map(|((pattern, _), expr)| (pattern, expr)),
            ),
            right_curly,
        ))
        .map(|((_, expr), branches)| Match {
            expr: Box::new(expr),
            branches,
        })
        .parse(input)
}

#[derive(Debug)]
pub struct Block {
    pub stats: Vec<Stat>,
    pub expr: Option<Box<Expr>>,
}

fn block(mut input: Input) -> Parsed<ScopeId> {
    let scope = input.ctx.borrow_mut().add_scope(Scope {
        kind: ScopeKind::Incomplete,
        types: HashMap::new(),
        values: HashMap::new(),
        parent: Some(input.scope),
    });
    input.scope = scope;
    delimited(left_curly, many0(stat).and(opt(expr)), right_curly)
        .map(|(_, expr)| {
            let expr = expr.unwrap_or_else(|| Expr {
                kind: ExprKind::Product(Product { fields: Vec::new() }),
                ty: input.ctx.borrow_mut().add_unit(),
            });
            input.ctx.borrow_mut().scopes[scope].kind = ScopeKind::Block(expr);
            scope
        })
        .parse(input.clone())
}

enum Part {
    Call(Call),
    Field(String),
}

fn dot_expr(input: Input) -> Parsed<Expr> {
    let convert = |expr, call: Call| {
        ExprKind::Call(Call {
            func: call.func.clone(),
            arg: Box::new(match *call.arg {
                Expr {
                    kind: ExprKind::Product(p),
                    ..
                } => ExprKind::Product(Product {
                    fields: [expr]
                        .into_iter()
                        .chain(p.fields.into_iter().map(|(_, field)| field))
                        .enumerate()
                        .map(|(i, field)| (i.to_string(), field))
                        .collect(),
                })
                .infer(&input),
                e => ExprKind::Product(Product {
                    fields: vec![("0".to_string(), expr), ("1".to_string(), e)],
                })
                .infer(&input),
            }),
        })
        .infer(&input)
    };

    let part = alt((call.map(Part::Call), ident.or(int).map(Part::Field)));

    _expr
        .and(dot)
        .and(separated_list1(dot, part))
        .map(|((base, _), parts)| {
            parts.into_iter().fold(base, |expr, part| match part {
                Part::Call(call) => convert(expr, call),
                Part::Field(field) => ExprKind::Access(Box::new(expr), field).infer(&input),
            })
        })
        .parse(input.clone())
}

fn _expr(input: Input) -> Parsed<Expr> {
    alt((
        call.map(ExprKind::Call),
        ident.map(ExprKind::Variable),
        int.map(ExprKind::Int),
        decimal.map(ExprKind::Decimal),
    ))
    .map(|x| x.infer(&input))
    .parse(input.clone())
}

fn expr(input: Input) -> Parsed<Expr> {
    alt((
        alt((
            block.map(ExprKind::Block),
            product(expr).map(ExprKind::Product),
            variant(expr).map(ExprKind::Variant),
            lambda.map(ExprKind::Lambda),
            pmatch.map(ExprKind::Match),
        ))
        .map(|x| x.infer(&input)),
        dot_expr,
        _expr,
    ))
    .parse(input.clone())
}

#[derive(Debug)]
pub struct ValueDecl {
    pub ident: String,
    pub expr: Expr,
}

fn vdecl(input: Input) -> Parsed<()> {
    let (input, value) = ident
        .and(equals)
        .and(expr)
        .and(semicolon)
        .map(|(((ident, _), expr), _)| ValueDecl { ident, expr })
        .parse(input)?;
    input.ctx.borrow_mut().scopes[input.scope]
        .values
        .insert(value.ident, value.expr);
    Ok((input, ()))
}

fn combined(input: Input) -> Parsed<()> {
    let (input, (ty, value)) = ident
        .and(colon)
        .and(ty)
        .and(equals)
        .and(expr)
        .and(semicolon)
        .map(|(((((ident, _), ty), _), expr), _)| {
            (
                TypeDecl {
                    ident: ident.clone(),
                    ty,
                },
                ValueDecl { ident, expr },
            )
        })
        .parse(input)?;
    input.ctx.borrow_mut().scopes[input.scope]
        .types
        .insert(ty.ident, ty.ty);
    input.ctx.borrow_mut().scopes[input.scope]
        .values
        .insert(value.ident, value.expr);
    Ok((input, ()))
}

#[derive(Debug)]
pub enum Stat {
    Type(TypeDecl),
    Value(ValueDecl),
}

fn fdecl(input: Input) -> Parsed<()> {
    let Some(Token::Identifier(ident)) = input.tokens.first().cloned() else {
        return IResult::Err(Err::Error(Error {
            kind: ErrorKind::MissingIdent,
            next: None,
        }));
    };
    let (input, value) = many1(
        keyword(ident.clone())
            .and(pattern)
            .and(equals)
            .and(expr)
            .and(semicolon)
            .map(|((((_, pattern), _), expr), _)| (pattern, expr)),
    )
    .map(|branches| ValueDecl {
        ident: ident.clone(),
        expr: ExprKind::Lambda(Lambda {
            param: "x".to_string(),
            body: Box::new(
                ExprKind::Match(Match {
                    expr: Box::new(ExprKind::Variable("x".to_string()).infer(&input)),
                    branches,
                })
                .infer(&input),
            ),
        })
        .infer(&input),
    })
    .parse(input.clone())?;
    input.ctx.borrow_mut().scopes[input.scope]
        .values
        .insert(value.ident, value.expr);
    Ok((input, ()))
}

fn stat(input: Input) -> Parsed<()> {
    alt((fdecl, tdecl, vdecl, combined))(input)
}

fn module(parent: Option<ScopeId>) -> impl Fn(Input) -> Parsed<ScopeId> {
    move |mut input| {
        let scope = input.ctx.borrow_mut().add_scope(Scope {
            kind: ScopeKind::Module,
            values: HashMap::new(),
            types: HashMap::new(),
            parent,
        });
        input.scope = scope;
        many0(stat).map(|_| scope).parse(input)
    }
}

pub fn parse_tokens(tokens: &[Token]) -> Option<Context> {
    let ctx = Context::default();
    let input = Input {
        ctx: Rc::new(RefCell::new(ctx)),
        scope: 0,
        tokens,
    };
    let (input, root) = module(None).parse(input).ok()?;
    println!("{root}");
    Some(input.ctx.take())
}
