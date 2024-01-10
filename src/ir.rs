use std::collections::{HashMap, VecDeque};

use crate::parse::{
    Call, Context, Expr, ExprKind, Lambda, Match, Product, Scope, ScopeKind, Sum, Type, TypeId,
    Variant,
};

pub fn prune_incomplete_blocks(ctx: &mut Context) {
    ctx.scopes
        .retain(|scope| !matches!(scope.kind, ScopeKind::IncompleteBlock))
}

/*
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
*/

pub fn infer(ctx: &mut Context) {
    let mut types: VecDeque<usize> = ctx
        .scopes
        .iter()
        .flat_map(|scope| {
            scope
                .values
                .iter()
                .map(|v| {
                    scope
                        .find_type(ctx, v.0)
                        .expect(&format!("Missing type for value {}", v.0))
                })
                .collect::<VecDeque<usize>>()
        })
        .collect();

    // Type lookup for values
    ctx.scopes.iter_mut().for_each(|scope| {
        scope.values.iter_mut().for_each(|value| {
            value.1.ty = types.pop_front().unwrap();
        })
    })
}
