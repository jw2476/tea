use lex::tokenize;
use parse::parse_tokens;

use crate::opt::remove_redundant_blocks;

mod lex;
mod opt;
mod parse;

fn main() {
    let file = String::from_utf8(std::fs::read("test.tea").unwrap()).unwrap();
    let tokens = tokenize(&file);
    let mut ctx = parse_tokens(&tokens).unwrap();
    remove_redundant_blocks(&mut ctx);
    println!("{ctx:#?}");
    println!("{}", ctx.display());
}
