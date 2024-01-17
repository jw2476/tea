use crate::parse::{BlockId, Context, Inst, Opcode, Operand};

pub fn remove_redundant_blocks(ctx: &mut Context) {
    let subs = ctx
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, block)| block.insts.len() == 1)
        .filter(|(_, block)| {
            matches!(ctx.insts[block.insts[0].0].op, Opcode::Return)
                && matches!(ctx.insts[block.insts[0].0].args[0], Operand::Block(_))
        })
        .map(|(i, block)| {
            (
                BlockId(i),
                ctx.insts[block.insts[0].0].args[0].to_block().unwrap(),
            )
        })
        .collect::<Vec<(BlockId, BlockId)>>();
    for (replace, with) in subs {
        println!("Replacing {:?} {:?}", replace, with);
        ctx.blocks.swap(replace.0, with.0);
    }
}
