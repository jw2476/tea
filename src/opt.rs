use std::collections::HashMap;

use crate::parse::{
    Block, BlockId, Context, Inst, InstId, Opcode, Operand, StringId, Symbol, Type, TypeId,
};

pub fn remove_redundant_blocks(ctx: &mut Context) {
    let subs = ctx
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, block)| block.insts.len() == 2)
        .filter(|(_, block)| {
            matches!(ctx.insts[block.insts[0].0].op, Opcode::As)
                && matches!(ctx.insts[block.insts[0].0].args[1], Operand::Block(_))
        })
        .map(|(i, block)| {
            (
                BlockId(i),
                ctx.insts[block.insts[0].0].args[1].to_block().unwrap(),
            )
        })
        .collect::<Vec<(BlockId, BlockId)>>();
    for (replace, with) in subs {
        let inst = ctx.blocks[replace.0].insts[0];
        ctx.blocks[with.0].insts.push(inst);
        println!("Replacing {:?} {:?}", replace, with);
        ctx.blocks.swap(replace.0, with.0);
    }
}

/*
pub fn resolve(ctx: &mut Context) {
    ctx.insts = ctx
        .insts
        .iter()
        .map(|inst| match inst.op {
            Opcode::Call => Inst {
                op: Opcode::Call,
                args: vec![
                    ctx.blocks[0]
                        .symbols
                        .get(&inst.args[0].to_ident().unwrap())
                        .expect("Function not found")
                        .value
                        .expect("Using type as value"),
                    inst.args[1],
                ],
            },
            _ => inst.clone(),
        })
        .collect();
}
*/

fn infer_block_args(ctx: &mut Context) {
    let symbols = ctx.symbols().collect::<Vec<(StringId, Symbol)>>();
    let lookup = |ctx: &Context, block: BlockId| {
        symbols
            .iter()
            .find(|(_, sy)| {
                sy.value
                    .and_then(|v| v.to_block().map(|b| b == block))
                    .unwrap_or_default()
            })
            .and_then(|(_, sy)| ctx.types[sy.ty?.0].clone().as_fn().map(|f| f.0))
    };

    let args = ctx
        .blocks
        .iter()
        .enumerate()
        .map(|(i, _)| lookup(ctx, BlockId(i)))
        .collect::<Vec<Option<TypeId>>>();

    ctx.blocks
        .iter_mut()
        .enumerate()
        .for_each(|(i, block)| block.arg = args[i]);
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ConstraintValue {
    Value(InstId),
    Arg(BlockId),
}

#[derive(Debug, Clone)]
pub enum ConstraintExpr {
    Var(ConstraintValue),
    //Construct(StringId, Vec<ConstraintExpr>),
    Get(Box<ConstraintExpr>, StringId),
    Labelled(Vec<(StringId, ConstraintExpr)>),
    Conrete(TypeId),
}

#[derive(Debug, Clone)]
pub struct Constraint(ConstraintExpr, ConstraintExpr);

fn constrain_expr(ctx: &Context, block: BlockId, oper: Operand) -> ConstraintExpr {
    match oper {
        Operand::Value(id) => ConstraintExpr::Var(ConstraintValue::Value(id)),
        Operand::Arg => {
            if let Some(ty) = ctx.blocks[block.0].arg {
                ConstraintExpr::Conrete(ty)
            } else {
                ConstraintExpr::Var(ConstraintValue::Arg(block))
            }
        }
        _ => todo!(),
    }
}

fn constrain(ctx: &Context, block: BlockId, id: InstId) -> Vec<Constraint> {
    let inst = ctx.insts[id.0].clone();
    let lhs = ConstraintExpr::Var(ConstraintValue::Value(id));
    match inst.op {
        Opcode::Get => vec![Constraint(
            lhs,
            ConstraintExpr::Get(
                Box::new(constrain_expr(ctx, block, inst.args[0])),
                inst.args[1].to_ident().unwrap(),
            ),
        )],
        Opcode::As => vec![Constraint(
            lhs,
            ConstraintExpr::Conrete(inst.args[0].to_ty().unwrap()),
        )],
        Opcode::Int => vec![Constraint(lhs, ConstraintExpr::Conrete(TypeId::INT))],
        Opcode::Product => vec![Constraint(
            lhs,
            ConstraintExpr::Labelled(
                inst.args
                    .chunks_exact(2)
                    .map(|x| x.try_into().unwrap())
                    .map(|[label, oper]: [Operand; 2]| {
                        (label.to_ident().unwrap(), constrain_expr(ctx, block, oper))
                    })
                    .collect(),
            ),
        )],
        _ => Vec::new(),
    }
}

fn unify(con: Constraint, subs: &mut HashMap<ConstraintValue, ConstraintExpr>) {
    match (con.0, con.1) {
        (ConstraintExpr::Var(lhs), ConstraintExpr::Var(rhs)) if lhs == rhs => {}
        (ConstraintExpr::Var(var), rhs) => {
            if let Some(sub) = subs.get(&var) {
                unify(Constraint(sub.clone(), rhs), subs);
            } else {
                subs.insert(var, rhs);
            }
        }
        (lhs, ConstraintExpr::Var(var)) => {
            if let Some(sub) = subs.get(&var) {
                unify(Constraint(sub.clone(), lhs), subs);
            } else {
                subs.insert(var, lhs);
            }
        }
        x => {
            println!("Skipped: {:?}", x);
        }
    }
}

fn get_block(ctx: &Context, inst: InstId) -> BlockId {
    ctx.blocks
        .iter()
        .enumerate()
        .find_map(|(i, block)| block.insts.iter().find(|i| **i == inst).map(|_| BlockId(i)))
        .unwrap()
}

pub fn infer(ctx: &mut Context) {
    infer_block_args(ctx);

    let constraints: Vec<Constraint> = (0..ctx.insts.len())
        .flat_map(|i| constrain(ctx, get_block(ctx, InstId(i)), InstId(i)))
        .collect();

    let mut subs = HashMap::new();
    constraints
        .into_iter()
        .for_each(|con| unify(con, &mut subs));

    println!("{:?}", subs);
}
