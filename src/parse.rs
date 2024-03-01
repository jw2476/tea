use std::marker::PhantomData;

use crate::lex::Token;

#[derive(Clone, Copy, Debug)]
pub struct Ctx<'a> {
    tokens: &'a [Token],
}

impl Ctx<'_> {
    pub fn pop(self) -> Self {
        Ctx {
            tokens: &self.tokens[1..],
        }
    }

    pub fn eat(self, token: &Token) -> Parsed<Self, ()> {
        match self.tokens.first() {
            Some(t) if t == token => Parsed::Some(self.pop(), ()),
            _ => Parsed::None(self),
        }
    }

    pub fn ident(self) -> Parsed<Self, String> {
        match self.tokens.first() {
            Some(Token::Identifier(ident)) => Parsed::Some(self.pop(), ident.clone()),
            _ => Parsed::None(self),
        }
    }

    pub fn int(self) -> Parsed<Self, String> {
        match self.tokens.first() {
            Some(Token::Integer(ident)) => Parsed::Some(self.pop(), ident.clone()),
            _ => Parsed::None(self),
        }
    }
}

pub enum Parsed<I, A> {
    Some(I, A),
    None(I),
}

impl<I, A> Parsed<I, A> {
    pub fn opt(self) -> (I, Option<A>) {
        match self {
            Self::Some(input, a) => (input, Some(a)),
            Self::None(input) => (input, None),
        }
    }
}

pub trait Parser<I, A> {
    fn parse(&mut self, input: I) -> Parsed<I, A>;

    fn map<B, G: Fn(A) -> B>(self, g: G) -> Map<Self, G, A>
    where
        Self: Sized,
    {
        Map {
            f: self,
            g,
            phantom: PhantomData,
        }
    }

    fn and<B, G: Parser<I, B>>(self, g: G) -> And<Self, G>
    where
        Self: Sized,
    {
        And { f: self, g }
    }

    fn silent_and<B, G: Parser<I, B>>(self, g: G) -> SilentAnd<Self, G, B>
    where
        Self: Sized,
    {
        SilentAnd {
            f: self,
            g,
            phantom: PhantomData,
        }
    }

    fn or<G: Parser<I, A>>(self, g: G) -> Or<Self, G>
    where
        Self: Sized,
    {
        Or { f: self, g }
    }
}

pub struct Map<F, G, A> {
    f: F,
    g: G,
    phantom: PhantomData<A>,
}

impl<I, A, B, F: Parser<I, A>, G: Fn(A) -> B> Parser<I, B> for Map<F, G, A> {
    fn parse(&mut self, input: I) -> Parsed<I, B> {
        match self.f.parse(input) {
            Parsed::Some(input, a) => Parsed::Some(input, (self.g)(a)),
            Parsed::None(input) => Parsed::None(input),
        }
    }
}

pub struct And<F, G> {
    f: F,
    g: G,
}

impl<I, A, B, F, G> Parser<I, (A, B)> for And<F, G>
where
    I: Clone,
    F: Parser<I, A>,
    G: Parser<I, B>,
{
    fn parse(&mut self, input: I) -> Parsed<I, (A, B)> {
        match self.f.parse(input.clone()) {
            Parsed::Some(i, a) => match self.g.parse(i) {
                Parsed::Some(i, b) => Parsed::Some(i, (a, b)),
                Parsed::None(i) => Parsed::None(input),
            },
            Parsed::None(i) => Parsed::None(input),
        }
    }
}

pub struct SilentAnd<F, G, B> {
    f: F,
    g: G,
    phantom: PhantomData<B>,
}

impl<I, A, B, F, G> Parser<I, A> for SilentAnd<F, G, B>
where
    I: Clone,
    F: Parser<I, A>,
    G: Parser<I, B>,
{
    fn parse(&mut self, input: I) -> Parsed<I, A> {
        match self.f.parse(input.clone()) {
            Parsed::Some(i, a) => match self.g.parse(i) {
                Parsed::Some(i, b) => Parsed::Some(i, a),
                Parsed::None(i) => Parsed::None(input),
            },
            Parsed::None(i) => Parsed::None(input),
        }
    }
}

pub struct Or<F, G> {
    f: F,
    g: G,
}

impl<I, A, F, G> Parser<I, A> for Or<F, G>
where
    F: Parser<I, A>,
    G: Parser<I, A>,
    I: Clone,
{
    fn parse(&mut self, input: I) -> Parsed<I, A> {
        match self.f.parse(input.clone()) {
            Parsed::None(_) => self.g.parse(input),
            x => x,
        }
    }
}

fn delimited<I, A, B, C, F, G, H>(
    mut left: F,
    mut value: G,
    mut right: H,
) -> impl FnMut(I) -> Parsed<I, B>
where
    F: Parser<I, A>,
    G: Parser<I, B>,
    H: Parser<I, C>,
    I: Clone,
{
    move |input: I| match left.parse(input.clone()) {
        Parsed::Some(i, _) => match value.parse(i) {
            Parsed::Some(i, value) => match right.parse(i) {
                Parsed::Some(i, _) => Parsed::Some(i, value),
                Parsed::None(_) => Parsed::None(input),
            },
            Parsed::None(_) => Parsed::None(input),
        },
        Parsed::None(_) => Parsed::None(input),
    }
}

fn many0<I, A, F>(mut f: F) -> impl FnMut(I) -> Parsed<I, Vec<A>>
where
    F: Parser<I, A>,
{
    move |input: I| {
        let mut parsed = Vec::new();
        let mut input = input;
        loop {
            input = match f.parse(input) {
                Parsed::Some(input, value) => {
                    parsed.push(value);
                    input
                }
                Parsed::None(input) => return Parsed::Some(input, parsed),
            }
        }
    }
}

fn many1<I, A, F>(mut f: F) -> impl FnMut(I) -> Parsed<I, Vec<A>>
where
    F: Parser<I, A>,
{
    move |input: I| match f.parse(input) {
        Parsed::Some(mut input, value) => {
            let mut parsed = vec![value];
            loop {
                input = match f.parse(input) {
                    Parsed::Some(input, value) => {
                        parsed.push(value);
                        input
                    }
                    Parsed::None(input) => return Parsed::Some(input, parsed),
                }
            }
        }
        Parsed::None(input) => Parsed::None(input),
    }
}

fn separated_list0<I, A, B, F, G>(
    mut value: F,
    mut separator: G,
) -> impl FnMut(I) -> Parsed<I, Vec<A>>
where
    F: Parser<I, A>,
    G: Parser<I, B>,
{
    move |input: I| {
        let (mut input, item) = match value.parse(input) {
            Parsed::Some(input, item) => (input, item),
            Parsed::None(input) => return Parsed::Some(input, Vec::new()),
        };

        let mut items = vec![item];

        loop {
            input = match separator.parse(input) {
                Parsed::None(input) => return Parsed::Some(input, items),
                Parsed::Some(input, _) => match value.parse(input) {
                    Parsed::None(input) => return Parsed::Some(input, items),
                    Parsed::Some(input, item) => {
                        items.push(item);
                        input
                    }
                },
            }
        }
    }
}

fn separated_list1<I, A, B, F, G>(
    mut value: F,
    mut separator: G,
) -> impl FnMut(I) -> Parsed<I, Vec<A>>
where
    F: Parser<I, A>,
    G: Parser<I, B>,
{
    move |input: I| {
        let (mut input, item) = match value.parse(input) {
            Parsed::Some(input, item) => (input, item),
            Parsed::None(input) => return Parsed::None(input),
        };

        let mut items = vec![item];

        loop {
            input = match separator.parse(input) {
                Parsed::None(input) => return Parsed::Some(input, items),
                Parsed::Some(input, _) => match value.parse(input) {
                    Parsed::None(input) => return Parsed::Some(input, items),
                    Parsed::Some(input, item) => {
                        items.push(item);
                        input
                    }
                },
            }
        }
    }
}

fn opt<I, A, F: Parser<I, A>>(mut f: F) -> impl FnMut(I) -> Parsed<I, Option<A>> {
    move |input: I| match f.parse(input) {
        Parsed::Some(input, a) => Parsed::Some(input, Some(a)),
        Parsed::None(input) => Parsed::Some(input, None),
    }
}

impl<'a> Parser<Ctx<'a>, ()> for Token {
    fn parse(&mut self, input: Ctx<'a>) -> Parsed<Ctx<'a>, ()> {
        input.eat(self)
    }
}

impl<I, A, T: FnMut(I) -> Parsed<I, A>> Parser<I, A> for T {
    fn parse(&mut self, input: I) -> Parsed<I, A> {
        self(input)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumType {
    pub variants: Vec<(String, Type)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductType {
    pub fields: Vec<(String, Type)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrimitiveType {
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
    Never,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Sum(SumType),
    Product(ProductType),
    Function(Box<Type>, Box<Type>),
    Named(String, Vec<Type>),
    Generic(String),
    Primitive(PrimitiveType),
}

fn keyword<'a>(text: &'a str) -> impl Parser<Ctx<'a>, ()> {
    move |input: Ctx<'a>| input.eat(&Token::Identifier(text.to_owned()))
}

fn ident(input: Ctx) -> Parsed<Ctx, String> {
    input.ident()
}

fn int(input: Ctx) -> Parsed<Ctx, String> {
    input.int()
}

fn tuple_ty(input: Ctx) -> Parsed<Ctx, ProductType> {
    delimited(
        Token::LeftRound,
        separated_list0(ty, Token::Comma),
        Token::RightRound,
    )
    .map(|fields| ProductType {
        fields: fields
            .into_iter()
            .enumerate()
            .map(|(i, field)| (i.to_string(), field))
            .collect(),
    })
    .parse(input)
}

fn product_ty(input: Ctx) -> Parsed<Ctx, ProductType> {
    delimited(
        Token::LeftCurly,
        separated_list0(ident.and(ty), Token::Comma),
        Token::RightCurly,
    )
    .map(|fields| ProductType { fields })
    .parse(input)
}

fn sum_ty(input: Ctx) -> Parsed<Ctx, SumType> {
    delimited(
        Token::LeftSquare,
        separated_list0(ident.and(ty), Token::Comma),
        Token::RightSquare,
    )
    .map(|variants| SumType { variants })
    .parse(input)
}

fn function_ty(input: Ctx) -> Parsed<Ctx, Type> {
    arg_ty
        .silent_and(Token::Arrow)
        .and(ty)
        .map(|(arg, ret)| Type::Function(Box::new(arg), Box::new(ret)))
        .parse(input)
}

fn primitive_ty(input: Ctx) -> Parsed<Ctx, Type> {
    (keyword("u8").map(|_| PrimitiveType::U8))
        .or(keyword("u16").map(|_| PrimitiveType::U16))
        .or(keyword("u32").map(|_| PrimitiveType::U32))
        .or(keyword("u64").map(|_| PrimitiveType::U64))
        .or(keyword("usize").map(|_| PrimitiveType::USize))
        .or(keyword("i8").map(|_| PrimitiveType::I8))
        .or(keyword("i16").map(|_| PrimitiveType::I16))
        .or(keyword("i32").map(|_| PrimitiveType::I32))
        .or(keyword("i64").map(|_| PrimitiveType::I64))
        .or(keyword("isize").map(|_| PrimitiveType::ISize))
        .or(keyword("f32").map(|_| PrimitiveType::F32))
        .or(keyword("f64").map(|_| PrimitiveType::F64))
        .or((Token::Bang).map(|_| PrimitiveType::Never))
        .map(Type::Primitive)
        .parse(input)
}

fn ty_common(input: Ctx, arg: bool) -> Parsed<Ctx, Type> {
    fn function(input: Ctx, arg: bool) -> Parsed<Ctx, Type> {
        if arg {
            delimited(Token::LeftRound, function_ty, Token::RightRound).parse(input)
        } else {
            function_ty.parse(input)
        }
    }

    (|input| function(input, arg))
        .or(tuple_ty.or(product_ty).map(Type::Product))
        .or(sum_ty.map(Type::Sum))
        .or(primitive_ty)
        .or(ident
            .and(generics)
            .map(|(ident, generics)| Type::Named(ident, generics)))
        .or(Token::Tick
            .and(ident)
            .map(|(_, ident)| ident)
            .map(Type::Generic))
        .parse(input)
}

fn ty(input: Ctx) -> Parsed<Ctx, Type> {
    ty_common(input, false)
}

fn arg_ty(input: Ctx) -> Parsed<Ctx, Type> {
    ty_common(input, true)
}

fn tdecl(input: Ctx) -> Parsed<Ctx, Stat> {
    ident
        .and(generics)
        .silent_and(Token::DoubleColon)
        .and(ty)
        .silent_and(Token::Semicolon)
        .map(|((ident, generics), ty)| Stat::TDecl(ident, generics, ty))
        .parse(input)
}

#[derive(Clone, Debug)]
pub enum Pattern {
    Tuple(Vec<Pattern>),
    Product(Vec<(String, Pattern)>),
    Variant(String, Box<Pattern>),
    Variable(String),
    Int(String),
}

fn pattern(input: Ctx) -> Parsed<Ctx, Pattern> {
    (int.map(Pattern::Int))
        .or(delimited(
            Token::LeftRound,
            separated_list0(pattern, Token::Comma),
            Token::RightRound,
        )
        .map(Pattern::Tuple))
        .or(delimited(
            Token::LeftCurly,
            separated_list0(ident.and(pattern), Token::Comma),
            Token::RightCurly,
        )
        .map(Pattern::Product))
        .or(ident
            .and(delimited(
                Token::LeftSquare,
                opt(pattern),
                Token::RightSquare,
            ))
            .map(|(ident, pattern)| {
                Pattern::Variant(
                    ident,
                    Box::new(pattern.unwrap_or(Pattern::Tuple(Vec::new()))),
                )
            }))
        .or(ident.map(Pattern::Variable))
        .parse(input)
}

#[derive(Clone, Debug)]
pub enum Expr {
    Int(String),
    Identifier(String),
    Access(Box<Expr>, String),
    Lambda(Pattern, Box<Expr>),
    Block(Vec<Stat>, Option<Box<Expr>>),
    Tuple(Vec<Expr>),
    Product(Vec<(String, Expr)>),
    Variant(String, Box<Expr>),
    Call(String, Box<Expr>),
    Match(Box<Expr>, Vec<(Pattern, Expr)>),
    Unreachable,
}

fn block(input: Ctx) -> Parsed<Ctx, Expr> {
    delimited(
        Token::LeftCurly,
        many0(stat).and(opt(expr)),
        Token::RightCurly,
    )
    .map(|(stats, expr)| Expr::Block(stats, expr.map(Box::new)))
    .parse(input)
}

fn lambda(input: Ctx) -> Parsed<Ctx, (Pattern, Expr)> {
    pattern.silent_and(Token::Arrow).and(expr).parse(input)
}

fn pmatch(input: Ctx) -> Parsed<Ctx, Expr> {
    keyword("match")
        .and(expr)
        .and(delimited(
            Token::LeftCurly,
            separated_list0(lambda, Token::Comma),
            Token::RightCurly,
        ))
        .map(|((_, base), branches)| Expr::Match(Box::new(base), branches))
        .parse(input)
}

fn part(input: Ctx) -> Parsed<Ctx, Expr> {
    (lambda.map(|(arg, body)| Expr::Lambda(arg, Box::new(body))))
        .or(keyword("unreachable").map(|_| Expr::Unreachable))
        .or(int.map(Expr::Int))
        .or(delimited(
            Token::LeftRound,
            separated_list0(expr, Token::Comma),
            Token::RightRound,
        )
        .map(Expr::Tuple))
        .or(delimited(
            Token::LeftCurly,
            separated_list0(ident.and(expr), Token::Comma),
            Token::RightCurly,
        )
        .map(Expr::Product))
        .or(ident
            .and(delimited(Token::LeftSquare, expr, Token::RightSquare))
            .map(|(ident, expr)| Expr::Variant(ident, Box::new(expr))))
        .or(ident
            .and(delimited(
                Token::LeftRound,
                (separated_list0(expr, Token::Comma).map(Expr::Tuple)).or(expr),
                Token::RightRound,
            ))
            .map(|(ident, arg)| Expr::Call(ident, Box::new(arg))))
        .or(pmatch)
        .or(ident.map(Expr::Identifier))
        .or(block)
        .parse(input)
}

fn expr(input: Ctx) -> Parsed<Ctx, Expr> {
    fn merge(base: Expr, suffix: Expr) -> Expr {
        match suffix {
            Expr::Int(index) => Expr::Access(Box::new(base), index.clone()),
            Expr::Call(ident, box Expr::Tuple(args)) => Expr::Call(
                ident.clone(),
                Box::new(Expr::Tuple([base].into_iter().chain(args).collect())),
            ),
            Expr::Call(ident, arg) => Expr::Call(
                ident.clone(),
                Box::new(Expr::Tuple(vec![base, *arg.clone()])),
            ),
            _ => panic!("unallowed as suffixes"),
        }
    }

    separated_list1(part, Token::Dot)
        .map(|parts| parts[1..].iter().cloned().fold(parts[0].clone(), merge))
        .parse(input)
}

#[derive(Clone, Debug)]
pub enum Stat {
    TDecl(String, Vec<Type>, Type),
    VDecl(String, Vec<Type>, Type, Expr),
    Expr(Expr),
}
fn generics(input: Ctx) -> Parsed<Ctx, Vec<Type>> {
    opt(delimited(
        Token::LeftAngle,
        separated_list0(ty, Token::Comma),
        Token::RightAngle,
    ))
    .map(Option::unwrap_or_default)
    .parse(input)
}

fn stat(input: Ctx) -> Parsed<Ctx, Stat> {
    tdecl
        .or(vdecl)
        .or(expr.silent_and(Token::Semicolon).map(Stat::Expr))
        .parse(input)
}

fn vdecl(input: Ctx) -> Parsed<Ctx, Stat> {
    ident
        .and(generics)
        .silent_and(Token::Colon)
        .and(ty)
        .silent_and(Token::Equals)
        .and(expr)
        .silent_and(Token::Semicolon)
        .map(|(((ident, generics), ty), expr)| Stat::VDecl(ident, generics, ty, expr))
        .parse(input)
}

pub fn parse(tokens: &[Token]) -> Option<Vec<Stat>> {
    let input = Ctx { tokens };
    many0(tdecl.or(vdecl)).parse(input).opt().1
}
