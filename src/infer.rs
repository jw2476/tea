use crate::ast::{Expr, ExprId, Stat, TypeId, AST};

pub fn infer(ast: &AST) -> Vec<TypeId> {
    let mut tys = vec![None; ast.exprs.len()];

    /*
    // Top-level decls all have types
    ast.decls
        .values()
        .filter(|(_, value)| value.is_some())
        .for_each(|(ty, value)| tys[value.unwrap().0] = Some(*ty));

    // So do block-level decls
    ast.exprs
        .iter()
        .filter_map(|expr| match expr {
            Expr::Block(stats, _) => Some(stats),
            _ => None,
        })
        .for_each(|stats| {
            stats
                .iter()
                .filter_map(|stat| match stat {
                    Stat::VDecl(_, ty, value) => Some((ty, value)),
                    _ => None,
                })
                .for_each(|(ty, value)| tys[value.0] = Some(*ty))
        });
    */

    ast.decls
        .values()
        .filter(|(_, value)| value.is_some())
        .for_each(|(ty, value)| infer_value(ast, &mut tys, ty, value));

    println!("{:?}", tys);

    tys.into_iter().map(Option::unwrap).collect()
}
