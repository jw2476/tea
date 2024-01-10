use lex::tokenize;
use parse::parse_tokens;

use crate::ir::{infer, prune_incomplete_blocks};

mod ir;
mod lex;
mod parse;

fn main() {
    let file = String::from_utf8(std::fs::read("test.tea").unwrap()).unwrap();
    let tokens = tokenize(&file);
    let mut ctx = parse_tokens(&tokens).unwrap();
    prune_incomplete_blocks(&mut ctx);
    infer(&mut ctx);
    println!("{ctx:#?}");
}
