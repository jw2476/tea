use crate::parse::{self, Pattern, PrimtiveType};
use std::{collections::HashMap, ops::Index};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypeId(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StringId(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExprId(pub usize);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumType {
    pub variants: Vec<(StringId, TypeId)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductType {
    pub fields: Vec<(StringId, TypeId)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Sum(SumType),
    Product(ProductType),
    Function(TypeId, TypeId),
    Alias(StringId),
    Primitive(PrimtiveType),
    Infer(ExprId),
}

impl Type {
    pub fn sum(&self) -> Option<&SumType> {
        match self {
            Self::Sum(x) => Some(x),
            _ => None,
        }
    }

    pub fn product(&self) -> Option<&ProductType> {
        match self {
            Self::Product(x) => Some(x),
            _ => None,
        }
    }
    pub fn function(&self) -> Option<(TypeId, TypeId)> {
        match self {
            Self::Function(arg, ret) => Some((*arg, *ret)),
            _ => None,
        }
    }

    pub fn dealias<'a>(&'a self, ast: &'a AST) -> &'a Self {
        match self {
            Self::Alias(alias) => &ast[ast.decls.get(alias).unwrap().0],
            x => x,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Expr {
    Int(StringId),
    Identifier(StringId),
    Access(ExprId, StringId),
    Lambda(ExprId, Vec<Cond>),
    Block(Vec<Stat>, Option<ExprId>),
    Tuple(Vec<ExprId>),
    Product(Vec<(StringId, ExprId)>),
    Variant(StringId, ExprId),
    Call(StringId, ExprId),
    Match(ExprId, Vec<ExprId>),
    Arg,
    Unreachable,
}

#[derive(Clone, Debug)]
pub enum Stat {
    TDecl(StringId, TypeId),
    VDecl(StringId, TypeId, ExprId),
    Expr(ExprId),
}

#[derive(Clone, Debug)]
pub enum Cond {
    Variant(ExprId, StringId),
}

#[derive(Default, Debug, Clone)]
pub struct AST {
    pub strings: Vec<String>,
    pub types: Vec<Type>,
    pub exprs: Vec<Expr>,
    pub decls: HashMap<StringId, (TypeId, Option<ExprId>)>,
}

impl AST {
    fn new() -> AST {
        Self::default()
    }

    pub fn add_string(&mut self, x: String) -> StringId {
        match self.strings.iter().position(|s| &x == s) {
            Some(index) => StringId(index),
            None => {
                self.strings.push(x);
                StringId(self.strings.len() - 1)
            }
        }
    }

    pub fn add_type(&mut self, x: Type) -> TypeId {
        match self.types.iter().position(|s| &x == s) {
            Some(index) => TypeId(index),
            None => {
                self.types.push(x);
                TypeId(self.types.len() - 1)
            }
        }
    }

    pub fn add_expr(&mut self, x: Expr) -> ExprId {
        self.exprs.push(x);
        ExprId(self.exprs.len() - 1)
    }
}

impl Index<StringId> for AST {
    type Output = String;

    fn index(&self, index: StringId) -> &Self::Output {
        &self.strings[index.0]
    }
}

impl Index<TypeId> for AST {
    type Output = Type;

    fn index(&self, index: TypeId) -> &Self::Output {
        &self.types[index.0]
    }
}

impl Index<ExprId> for AST {
    type Output = Expr;

    fn index(&self, index: ExprId) -> &Self::Output {
        &self.exprs[index.0]
    }
}

fn desugar_pattern(ast: &mut AST, pattern: Pattern, value: ExprId) -> (Vec<Stat>, Vec<Cond>) {
    match pattern {
        Pattern::Tuple(patterns) => patterns
            .into_iter()
            .enumerate()
            .map(|(index, pattern)| {
                let index = ast.add_string(index.to_string());
                let value = ast.add_expr(Expr::Access(value, index));
                desugar_pattern(ast, pattern, value)
            })
            .fold(
                (Vec::new(), Vec::new()),
                |(mut stats_acc, mut conds_acc), (stats, conds)| {
                    stats_acc.extend(stats);
                    conds_acc.extend(conds);
                    (stats_acc, conds_acc)
                },
            ),
        Pattern::Product(patterns) => patterns
            .into_iter()
            .map(|(index, pattern)| {
                let index = ast.add_string(index.to_string());
                let value = ast.add_expr(Expr::Access(value, index));
                desugar_pattern(ast, pattern, value)
            })
            .fold(
                (Vec::new(), Vec::new()),
                |(mut stats_acc, mut conds_acc), (stats, conds)| {
                    stats_acc.extend(stats);
                    conds_acc.extend(conds);
                    (stats_acc, conds_acc)
                },
            ),
        Pattern::Variant(ident, pattern) => {
            let ident = ast.add_string(ident);
            let inner = ast.add_expr(Expr::Access(value, ident));
            let (stats, mut conds) = desugar_pattern(ast, *pattern, inner);
            conds.push(Cond::Variant(value, ident));
            (stats, conds)
        }
        Pattern::Variable(ident) => {
            let ident = ast.add_string(ident);
            let ty = ast.add_type(Type::Infer(value));
            (vec![Stat::VDecl(ident, ty, value)], vec![])
        }
        Pattern::Int(int) => todo!(),
    }
}

fn flatten_lambda(ast: &mut AST, pattern: parse::Pattern, body: parse::Expr) -> Expr {
    let arg = ast.add_expr(Expr::Arg);
    let pattern = desugar_pattern(ast, pattern, arg);
    let body = match body {
        parse::Expr::Block(stats, expr) => {
            let expr = expr.map(|expr| flatten_expr(ast, *expr));
            Expr::Block(
                pattern
                    .0
                    .into_iter()
                    .chain(stats.into_iter().map(|stat| flatten_stat(ast, stat)))
                    .collect(),
                expr,
            )
        }
        expr => Expr::Block(pattern.0, Some(flatten_expr(ast, expr))),
    };
    Expr::Lambda(ast.add_expr(body), pattern.1)
}

fn flatten_expr(ast: &mut AST, expr: parse::Expr) -> ExprId {
    let expr = match expr {
        parse::Expr::Int(int) => Expr::Int(ast.add_string(int)),
        parse::Expr::Identifier(ident) => Expr::Identifier(ast.add_string(ident)),
        parse::Expr::Access(expr, accessor) => {
            let expr = flatten_expr(ast, *expr);
            let accessor = ast.add_string(accessor);
            Expr::Access(expr, accessor)
        }
        parse::Expr::Lambda(pattern, body) => flatten_lambda(ast, pattern, *body),
        parse::Expr::Call(ident, arg) => {
            let ident = ast.add_string(ident);
            let arg = flatten_expr(ast, *arg);
            Expr::Call(ident, arg)
        }
        parse::Expr::Block(stats, expr) => {
            let stats = stats
                .into_iter()
                .map(|stat| flatten_stat(ast, stat))
                .collect();
            let expr = expr.map(|expr| flatten_expr(ast, *expr));
            Expr::Block(stats, expr)
        }
        parse::Expr::Tuple(exprs) => Expr::Tuple(
            exprs
                .into_iter()
                .map(|expr| flatten_expr(ast, expr))
                .collect(),
        ),
        parse::Expr::Product(fields) => Expr::Product(
            fields
                .into_iter()
                .map(|(ident, expr)| {
                    let ident = ast.add_string(ident);
                    let expr = flatten_expr(ast, expr);
                    (ident, expr)
                })
                .collect(),
        ),
        parse::Expr::Variant(ident, expr) => {
            let ident = ast.add_string(ident);
            let expr = flatten_expr(ast, *expr);
            Expr::Variant(ident, expr)
        }
        parse::Expr::Match(base, branches) => {
            let base = flatten_expr(ast, *base);
            let branches = branches
                .into_iter()
                .map(|(pattern, body)| {
                    let expr = flatten_lambda(ast, pattern, body);
                    ast.add_expr(expr)
                })
                .collect();

            Expr::Match(base, branches)
        }
        parse::Expr::Unreachable => Expr::Unreachable,
    };
    ast.add_expr(expr)
}

fn flatten_stat(ast: &mut AST, stat: parse::Stat) -> Stat {
    match stat {
        parse::Stat::TDecl(ident, ty) => {
            let ident = ast.add_string(ident);
            let ty = flatten_type(ast, ty);
            Stat::TDecl(ident, ty)
        }
        parse::Stat::VDecl(ident, ty, value) => {
            let ident = ast.add_string(ident);
            let ty = flatten_type(ast, ty);
            let value = flatten_expr(ast, value);
            Stat::VDecl(ident, ty, value)
        }
        parse::Stat::Expr(expr) => {
            let expr = flatten_expr(ast, expr);
            Stat::Expr(expr)
        }
    }
}

fn flatten_type(ast: &mut AST, ty: parse::Type) -> TypeId {
    let ty = match ty {
        parse::Type::Sum(parse::SumType { variants }) => Type::Sum(SumType {
            variants: variants
                .into_iter()
                .map(|(ident, ty)| {
                    let ident = ast.add_string(ident);
                    let ty = flatten_type(ast, ty);
                    (ident, ty)
                })
                .collect(),
        }),
        parse::Type::Product(parse::ProductType { fields }) => Type::Product(ProductType {
            fields: fields
                .into_iter()
                .map(|(ident, ty)| {
                    let ident = ast.add_string(ident);
                    let ty = flatten_type(ast, ty);
                    (ident, ty)
                })
                .collect(),
        }),
        parse::Type::Function(arg, ret) => {
            let arg = flatten_type(ast, *arg);
            let ret = flatten_type(ast, *ret);
            Type::Function(arg, ret)
        }
        parse::Type::Alias(ident) => Type::Alias(ast.add_string(ident.clone())),
        parse::Type::Primitive(ty) => Type::Primitive(ty),
    };
    ast.add_type(ty)
}

fn flatten_decl(ast: &mut AST, decl: parse::Stat) {
    let decl = flatten_stat(ast, decl);
    match decl {
        Stat::TDecl(ident, ty) => ast.decls.insert(ident, (ty, None)),
        Stat::VDecl(ident, ty, value) => ast.decls.insert(ident, (ty, Some(value))),
        _ => panic!(),
    };
}

pub fn flatten(decls: Vec<parse::Stat>) -> AST {
    let mut ast = AST::new();
    decls
        .into_iter()
        .for_each(|decl| flatten_decl(&mut ast, decl));
    ast
}
