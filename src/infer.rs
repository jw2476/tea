use std::collections::HashMap;

use crate::{
    ast::{Expr, ExprId, ProductType, Stat, StringId, SumType, Type, TypeId, AST},
    parse::PrimtiveType,
};

#[derive(Clone, Debug)]
pub struct InferState {
    arg_ty: Option<TypeId>,
    decls: HashMap<StringId, TypeId>,
}

pub fn get_ty(ast: &mut AST, expr: ExprId, types: &[Option<TypeId>], state: InferState) -> TypeId {
    if let Some(ty) = types[expr.0] {
        return ty;
    }

    match ast[expr].clone() {
        Expr::Call(ident, _) => {
            let func_ty = ast.decls.get(&ident).expect("missing decls").0;
            ast[func_ty].function().unwrap().1
        }
        Expr::Block(_, expr) => match expr {
            Some(expr) => get_ty(ast, expr, types, state),
            None => ast.add_type(Type::Product(ProductType { fields: Vec::new() })),
        },
        Expr::Int(_) => ast.add_type(Type::Primitive(PrimtiveType::I32)), // APInts
        Expr::Identifier(ident) => *state
            .decls
            .get(&ident)
            .or(ast.decls.get(&ident).map(|x| &x.0))
            .unwrap(),
        Expr::Access(base, accessor) => {
            let base_ty = get_ty(ast, base, types, state);
            match &ast[base_ty].dealias(ast) {
                Type::Product(ProductType { fields }) => {
                    fields
                        .iter()
                        .find(|(ident, _)| *ident == accessor)
                        .unwrap()
                        .1
                }
                Type::Sum(SumType { variants }) => {
                    variants
                        .iter()
                        .find(|(ident, _)| *ident == accessor)
                        .unwrap()
                        .1
                }
                _ => panic!("Trying to access a primitive type"),
            }
        }
        Expr::Tuple(elements) => {
            let element_types = elements
                .iter()
                .enumerate()
                .map(|(i, element)| {
                    (
                        ast.add_string(i.to_string()),
                        get_ty(ast, *element, types, state.clone()),
                    )
                })
                .collect();
            ast.add_type(Type::Product(ProductType {
                fields: element_types,
            }))
        }
        Expr::Product(elements) => {
            let element_types = elements
                .iter()
                .map(|(i, element)| (*i, get_ty(ast, *element, types, state.clone())))
                .collect();
            ast.add_type(Type::Product(ProductType {
                fields: element_types,
            }))
        }
        Expr::Arg => state.arg_ty.unwrap(),
        Expr::Unreachable => ast.add_type(Type::Primitive(PrimtiveType::Never)),
        x => todo!("{:?}", x),
    }
}

pub fn propagate(
    ast: &mut AST,
    ty: TypeId,
    expr: ExprId,
    types: &mut [Option<TypeId>],
    mut state: InferState,
) {
    if let Expr::Arg = ast[expr] {
        types[expr.0] = Some(state.arg_ty.unwrap());
        return;
    }

    types[expr.0] = Some(ty);

    match ast[expr].clone() {
        Expr::Call(ident, arg) => {
            let func_ty = ast.decls.get(&ident).expect("Missing decl").0;
            let param_ty = match ast[func_ty] {
                Type::Function(arg, _) => arg,
                _ => panic!("Calling {} which is not a function", ast[ident]),
            };
            propagate(ast, param_ty, arg, types, state);
        }
        Expr::Block(stats, expr) => {
            state = stats.iter().fold(state, |mut state, stat| {
                if let Stat::VDecl(ident, ty, value) = stat {
                    let ty = if let Type::Infer(_) = ast[*ty] {
                        get_ty(ast, *value, types, state.clone())
                    } else {
                        *ty
                    };
                    propagate(ast, ty, *value, types, state.clone());
                    state.decls.insert(*ident, ty);
                };
                state
            });

            if let Some(expr) = expr {
                propagate(ast, ty, expr, types, state)
            }
        }
        Expr::Tuple(elements) => match ast[ty].dealias(ast).clone() {
            Type::Product(ProductType { fields }) => fields
                .iter()
                .map(|(_, x)| x)
                .zip(elements)
                .for_each(|(ty, element)| propagate(ast, *ty, element, types, state.clone())),
            _ => panic!("Tuple doesn't have tuple type"),
        },
        Expr::Access(base, _) => {
            let base_ty = get_ty(ast, base, types, state.clone());
            propagate(ast, base_ty, base, types, state);
        }
        Expr::Product(fields) => {
            let field_types = match ast[ty].dealias(ast).clone() {
                Type::Product(ProductType { fields }) => fields,
                _ => panic!("Not a product type"),
            };
            fields.iter().for_each(|(ident, expr)| {
                let ty = field_types
                    .iter()
                    .find(|(i, _)| ident == i)
                    .map(|(_, ty)| ty)
                    .unwrap();
                propagate(ast, *ty, *expr, types, state.clone());
            })
        }
        Expr::Match(base, branches) => {
            let base_ty = get_ty(ast, base, types, state.clone());
            let branch_ty = ast.add_type(Type::Function(base_ty, ty));
            branches
                .into_iter()
                .for_each(|branch| propagate(ast, branch_ty, branch, types, state.clone()));
        }
        Expr::Identifier(_) | Expr::Int(_) | Expr::Arg | Expr::Unreachable => (),
        Expr::Lambda(body, _) => {
            let (arg, ret) = ast[ty].function().unwrap();
            state.arg_ty = Some(arg);
            propagate(ast, ret, body, types, state);
        }
        Expr::Variant(variant, expr) => {
            let variants = match &ast[ty].dealias(ast) {
                Type::Sum(SumType { variants }) => variants,
                _ => panic!("not a sum type"),
            };
            let ty = variants
                .iter()
                .find(|(ident, _)| *ident == variant)
                .expect("invalid variant")
                .1;
            propagate(ast, ty, expr, types, state);
        }
    }
}

pub fn infer(ast: &mut AST) -> Vec<TypeId> {
    let mut types = vec![None; ast.exprs.len()];
    // Top-level decls all have types
    ast.decls
        .clone()
        .values()
        .filter(|(_, value)| value.is_some())
        .for_each(|(ty, value)| {
            propagate(
                ast,
                *ty,
                value.unwrap(),
                &mut types,
                InferState {
                    arg_ty: None,
                    decls: HashMap::new(),
                },
            )
        });

    println!(
        "{:#?}",
        types.iter().map(|x| x.map(|x| &ast[x])).collect::<Vec<_>>()
    );

    println!(
        "{:#?}",
        types
            .iter()
            .enumerate()
            .filter(|(_, x)| x.is_none())
            .map(|(i, _)| &ast.exprs[i])
            .collect::<Vec<_>>()
    );

    println!("{:?}", ast.strings[11]);

    types.into_iter().map(Option::unwrap).collect()
}
