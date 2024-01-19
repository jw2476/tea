use ir::{
    arith::{addi, int},
    func::func,
    IR,
};
use lex::tokenize;
use parse::parse_tokens;

use crate::{
    ir::{func::ret, Type},
    opt::{infer, remove_redundant_blocks},
};

mod ir;
mod lex;
mod opt;
mod parse;

fn main() {
    /*
    let file = String::from_utf8(std::fs::read("test.tea").unwrap()).unwrap();
    let tokens = tokenize(&file);
    let mut ctx = parse_tokens(&tokens).unwrap();
    remove_redundant_blocks(&mut ctx);
    //resolve(&mut ctx);
    infer(&mut ctx);
    //println!("{ctx:#?}");
    println!("{}", ctx.display());
    */

    let mut ir = IR::new();
    let region = ir.new_region();
    let label = ir.add_string("test");
    let unit = ir.add_ty(Type::Product(Vec::new()));
    let block = ir.append_block(region, label, Some(unit));
    let one = ir.add_string("1");
    let two = ir.add_string("2");
    let int_ty = ir.add_ty(Type::I32);
    let lhs = int(&mut ir, block, one, Some(int_ty));
    let rhs = int(&mut ir, block, two, Some(int_ty));
    let int_pair_ty = ir.add_ty(Type::Product(vec![int_ty; 2]));
    let addi_ty = ir.add_ty(Type::Function(int_pair_ty, int_ty));
    let res = addi(&mut ir, block, lhs.into(), rhs.into(), Some(addi_ty));
    ret(&mut ir, block, res.into(), Some(int_ty));
    let main = ir.add_string("main");
    let main_ty = ir.add_ty(Type::Function(unit, int_ty));
    func(&mut ir, None, main, region, Some(main_ty));
    println!("{:?}", res);
    println!("{}", ir.display());
}
