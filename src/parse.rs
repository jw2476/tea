use std::{
    collections::HashMap,
    convert::Infallible,
    marker::PhantomData,
    ops::{FromResidual, RangeFrom, Try},
};

use crate::{
    ir::{func, types, BlockId, Decl, OpId, RegionId, StringId, Type, TypeId, Value, IR},
    lex::Token,
};

#[derive(Clone, Debug)]
pub struct Input<'a> {
    tokens: &'a [Token],
    ir: IR,
    symbols: HashMap<StringId, Value>,
}

pub enum Parsed<I, A> {
    Some(I, A),
    None(I),
}

impl<I, A> FromResidual for Parsed<I, A> {
    fn from_residual(residual: <Self as Try>::Residual) -> Self {
        match residual {
            Parsed::None(input) => Self::None(input),
            _ => panic!(),
        }
    }
}

impl<I, A> Try for Parsed<I, A> {
    type Output = (I, A);
    type Residual = Parsed<I, Infallible>;

    fn from_output(output: Self::Output) -> Self {
        Self::Some(output.0, output.1)
    }

    fn branch(self) -> std::ops::ControlFlow<Self::Residual, Self::Output> {
        match self {
            Self::Some(input, x) => std::ops::ControlFlow::Continue((input, x)),
            Self::None(input) => std::ops::ControlFlow::Break(Parsed::None(input)),
        }
    }
}

impl<I, A> Parsed<I, A> {
    fn to_opt(self) -> Option<(I, A)> {
        match self {
            Self::Some(input, value) => Some((input, value)),
            Self::None(_) => None,
        }
    }

    pub fn map<B, F: Fn(A) -> B>(self, f: F) -> Parsed<I, B> {
        match self {
            Self::Some(input, a) => Parsed::Some(input, f(a)),
            Self::None(input) => Parsed::None(input),
        }
    }
}

impl Input<'_> {
    fn slice(self, range: RangeFrom<usize>) -> Self {
        Self {
            tokens: &self.tokens[range],
            ir: self.ir,
            symbols: self.symbols,
        }
    }
}

pub trait Parser<I, A> {
    fn parse(&self, input: I) -> Parsed<I, A>;

    fn and<B, G: Parser<I, B>>(self, g: G) -> And<Self, G>
    where
        Self: Sized,
    {
        And { f: self, g }
    }

    fn or<G: Parser<I, A>>(self, g: G) -> Or<Self, G>
    where
        Self: Sized,
    {
        Or { f: self, g }
    }

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
}

pub struct And<F, G> {
    f: F,
    g: G,
}

impl<I, A, B, F, G> Parser<I, (A, B)> for And<F, G>
where
    F: Parser<I, A>,
    G: Parser<I, B>,
{
    fn parse(&self, input: I) -> Parsed<I, (A, B)> {
        let (input, a) = self.f.parse(input)?;
        let (input, b) = self.g.parse(input)?;
        Parsed::Some(input, (a, b))
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
    fn parse(&self, input: I) -> Parsed<I, A> {
        match self.f.parse(input.clone()) {
            Parsed::None(_) => self.g.parse(input),
            x => x,
        }
    }
}

pub struct Map<F, G, A> {
    f: F,
    g: G,
    phantom: PhantomData<A>,
}

impl<I, A, B, F, G> Parser<I, B> for Map<F, G, A>
where
    F: Parser<I, A>,
    G: Fn(A) -> B,
{
    fn parse(&self, input: I) -> Parsed<I, B> {
        match self.f.parse(input) {
            Parsed::Some(input, a) => Parsed::Some(input, (self.g)(a)),
            Parsed::None(input) => Parsed::None(input),
        }
    }
}

impl<I, A, T> Parser<I, A> for T
where
    T: Fn(I) -> Parsed<I, A>,
{
    fn parse(&self, input: I) -> Parsed<I, A> {
        self(input)
    }
}

pub fn apply<IA, IB, A, F>(f: F, value: IB) -> Apply<F, IB>
where
    F: Fn(IA, IB) -> Parsed<IA, A>,
{
    Apply { f, value }
}

pub struct Apply<F, IB> {
    f: F,
    value: IB,
}

impl<IA, IB, A, F> Parser<IA, A> for Apply<F, IB>
where
    F: Fn(IA, IB) -> Parsed<IA, A>,
    IB: Clone,
{
    fn parse(&self, input: IA) -> Parsed<IA, A> {
        (self.f)(input, self.value.clone())
    }
}

fn ident(mut input: Input) -> Parsed<Input, StringId> {
    match input.tokens.first().cloned() {
        Some(Token::Identifier(ident)) => {
            let id = input.ir.add_string(ident);
            Parsed::Some(input.slice(1..), id)
        }
        _ => Parsed::None(input),
    }
}

fn int(mut input: Input) -> Parsed<Input, StringId> {
    match input.tokens.first().cloned() {
        Some(Token::Integer(ident)) => {
            let id = input.ir.add_string(ident);
            Parsed::Some(input.slice(1..), id)
        }
        _ => Parsed::None(input),
    }
}

macro_rules! tok_parser {
    ($n:ident, $p: pat) => {
        fn $n(input: Input) -> Parsed<Input, ()> {
            match input.tokens.first().cloned() {
                Some($p) => Parsed::Some(input.slice(1..), ()),
                _ => Parsed::None(input),
            }
        }
    };
}

tok_parser!(double_colon, Token::DoubleColon);
tok_parser!(comma, Token::Comma);
tok_parser!(left_square, Token::LeftSquare);
tok_parser!(right_square, Token::RightSquare);
tok_parser!(left_curly, Token::LeftCurly);
tok_parser!(right_curly, Token::RightCurly);
tok_parser!(left_round, Token::LeftRound);
tok_parser!(right_round, Token::RightRound);
tok_parser!(arrow, Token::Arrow);
tok_parser!(semicolon, Token::Semicolon);
tok_parser!(equals, Token::Equals);
tok_parser!(pipe, Token::Pipe);
tok_parser!(wide_arrow, Token::WideArrow);
tok_parser!(dot, Token::Dot);
tok_parser!(colon, Token::Colon);
tok_parser!(underscore, Token::Underscore);
tok_parser!(tick, Token::Tick);
tok_parser!(left_angle, Token::LeftAngle);
tok_parser!(right_angle, Token::RightAngle);

fn delimited<I, A, B, C, F, G, H>(left: F, value: G, right: H) -> impl Fn(I) -> Parsed<I, B>
where
    F: Parser<I, A>,
    G: Parser<I, B>,
    H: Parser<I, C>,
    B: Clone,
{
    move |input: I| {
        let (input, _) = left.parse(input)?;
        let (input, x) = value.parse(input)?;
        right.parse(input).map(|_| x.clone())
    }
}

fn many0<I, A, F>(f: F) -> impl Fn(I) -> Parsed<I, Vec<A>>
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

fn separated_list0<I, A, B, F, G>(value: F, separator: G) -> impl Fn(I) -> Parsed<I, Vec<A>>
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

fn keyword<T: ToString>(word: T) -> impl Fn(Input) -> Parsed<Input, ()> {
    move |mut input: Input| {
        let id = input.ir.add_string(word.to_string());
        match ident.parse(input.clone()) {
            Parsed::Some(i, ident) if ident == id => Parsed::Some(i, ()),
            Parsed::Some(_, _) => Parsed::None(input),
            Parsed::None(_) => Parsed::None(input),
        }
    }
}

fn keyword_ty(input: Input) -> Parsed<Input, TypeId> {
    let (mut input, ty) = (keyword("u8").map(|_| Type::U8))
        .or(keyword("u16").map(|_| Type::U16))
        .or(keyword("u32").map(|_| Type::U32))
        .or(keyword("u64").map(|_| Type::U64))
        .or(keyword("usize").map(|_| Type::USize))
        .or(keyword("i8").map(|_| Type::I8))
        .or(keyword("i16").map(|_| Type::I16))
        .or(keyword("i32").map(|_| Type::I32))
        .or(keyword("i64").map(|_| Type::I64))
        .or(keyword("isize").map(|_| Type::ISize))
        .or(keyword("f32").map(|_| Type::F32))
        .or(keyword("f64").map(|_| Type::F64))
        .parse(input)?;
    let ty = input.ir.add_ty(ty);
    Parsed::Some(input, ty)
}

fn tuple_ty(input: Input) -> Parsed<Input, TypeId> {
    let (mut input, ty) = delimited(left_round, separated_list0(ty, comma), right_round)
        .parse(input)
        .map(|fields| Type::Product(fields.into_iter().map(Some).collect()))?;
    let id = input.ir.add_ty(ty);
    Parsed::Some(input, id)
}

fn sum_ty(input: Input) -> Parsed<Input, TypeId> {
    let (mut input, ty) = delimited(left_square, separated_list0(ty, pipe), right_square)
        .parse(input)
        .map(|variants| Type::Sum(variants.into_iter().map(Some).collect()))?;
    let id = input.ir.add_ty(ty);
    Parsed::Some(input, id)
}

fn function_ty(input: Input) -> Parsed<Input, TypeId> {
    let (mut input, ty) = arg
        .and(arrow)
        .and(ty)
        .parse(input)
        .map(|((arg, _), ret)| Type::Function(Some(arg), Some(ret)))?;
    let id = input.ir.add_ty(ty);
    Parsed::Some(input, id)
}

fn alias_ty(input: Input) -> Parsed<Input, TypeId> {
    let (mut i, ident) = ident.parse(input.clone())?;
    if let Some(ty) = i.ir.get_ty_alias(ident) {
        Parsed::Some(i, ty)
    } else {
        Parsed::None(input)
    }
}

fn arg(input: Input) -> Parsed<Input, TypeId> {
    delimited(left_round, function_ty, left_round)
        .or(tuple_ty)
        .or(sum_ty)
        .or(keyword_ty)
        .or(alias_ty)
        .parse(input)
}

fn ty(input: Input) -> Parsed<Input, TypeId> {
    function_ty
        .or(tuple_ty)
        .or(sum_ty)
        .or(keyword_ty)
        .or(alias_ty)
        .parse(input)
}

fn tdecl(input: Input) -> Parsed<Input, ()> {
    let (mut input, (ident, ty)) = ident
        .and(double_colon)
        .and(ty)
        .and(semicolon)
        .parse(input)
        .map(|(((ident, _), ty), _)| (ident, ty))?;
    input.ir.add_ty_alias(ident, ty);
    Parsed::Some(input, ())
}

fn variable(input: Input, symbols: HashMap<StringId, Value>) -> Parsed<Input, Value> {
    let (i, ident) = ident.parse(input.clone())?;
    match symbols.get(&ident) {
        Some(value) => Parsed::Some(i, *value),
        None => Parsed::None(input),
    }
}

fn access(input: Input, state: (BlockId, HashMap<StringId, Value>)) -> Parsed<Input, Value> {
    let (block, symbols) = state;
    let (mut input, (base, indices)) = apply(variable, symbols)
        .and(many0(dot.and(int).map(|(_, index)| index)))
        .parse(input)?;

    let (value, _) = indices
        .iter()
        .fold((base, base.ty(&input.ir)), |(value, ty), index| {
            let ty = ty.and_then(|ty| match &input.ir[ty] {
                Type::Product(fields) => fields[input.ir[*index].parse::<usize>().unwrap()],
                _ => panic!(),
            });
            let op = types::get(&mut input.ir, block, value, *index, ty);
            (Value::Op(op), ty)
        });
    Parsed::Some(input, value)
}

fn tuple_expr(
    input: Input,
    state: (RegionId, BlockId, HashMap<StringId, Value>),
) -> Parsed<Input, Value> {
    let (mut input, fields) = delimited(
        left_round,
        separated_list0(apply(expr, state.clone()), comma),
        right_round,
    )
    .parse(input)?;
    let ty = Type::Product(fields.iter().map(|field| field.ty(&input.ir)).collect());
    let ty = input.ir.add_ty(ty);
    let id = types::product(&mut input.ir, state.1, fields, Some(ty));
    Parsed::Some(input, Value::Op(id))
}

fn call(
    input: Input,
    state: (RegionId, BlockId, HashMap<StringId, Value>),
) -> Parsed<Input, Value> {
    let (mut input, (ident, arg)) = ident
        .and(delimited(
            left_round,
            apply(expr, state.clone()),
            right_round,
        ))
        .parse(input)?;
    let func = input.symbols.get(&ident).unwrap();
    let ret_ty = func.ty(&input.ir).and_then(|ty| match input.ir[ty] {
        Type::Function(_, ret) => ret,
        _ => None,
    });

    let id = func::call(&mut input.ir, state.1, *func, arg, ret_ty);
    Parsed::Some(input, Value::Op(id))
}

fn expr(
    input: Input,
    state: (RegionId, BlockId, HashMap<StringId, Value>),
) -> Parsed<Input, Value> {
    let (region, block, symbols) = state.clone();
    apply(access, (block, symbols.clone()))
        .or(apply(call, state.clone()))
        .or(apply(tuple_expr, state))
        .or(apply(variable, symbols))
        .parse(input)
}

fn fdecl(input: Input) -> Parsed<Input, ()> {
    let (mut input, (ident, ty, arg)) = ident
        .and(colon)
        .and(ty)
        .and(equals)
        .and(ident)
        .and(arrow)
        .parse(input)
        .map(|(((((ident, _), ty), _), arg), _)| (ident, ty, arg))?;
    let region = input.ir.new_region();
    let entry = input.ir.add_string("entry");
    let Type::Function(arg_ty, ret_ty) = input.ir[ty] else {
        panic!()
    };
    let block = input.ir.append_block(region, entry, arg_ty);
    let (mut input, (value, _)) = apply(
        expr,
        (region, block, HashMap::from([(arg, Value::Arg(block))])),
    )
    .and(semicolon)
    .parse(input)?;
    if let Some((value_ty, ret_ty)) = value
        .ty(&input.ir)
        .and_then(|value_ty| Some((value_ty, ret_ty?)))
    {
        assert_eq!(value_ty, ret_ty);
    }

    func::ret(&mut input.ir, block, value, ret_ty);
    let id = func::func(&mut input.ir, None, ident, region, Some(ty));
    input.symbols.insert(ident, Value::Op(id));
    Parsed::Some(input, ())
}

pub fn parse_tokens(tokens: &[Token]) -> Option<IR> {
    let input = Input {
        tokens,
        ir: IR::new(),
        symbols: HashMap::new(),
    };
    let (input, parsed) = many0(tdecl.or(fdecl)).parse(input).to_opt()?;
    println!("{:?}", parsed);
    //let (input, parsed) = many0(tdecl.or(vdecl)).parse(input).unwrap();
    Some(input.ir)
}
