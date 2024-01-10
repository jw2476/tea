use lex::tokenize;
use parse::parse_tokens;

mod lex;
mod parse;

fn main() {
    let file = String::from_utf8(std::fs::read("test.tea").unwrap()).unwrap();
    let tokens = tokenize(&file);
    let (mut ctx, decls) = parse_tokens(&tokens).unwrap();
    println!("{ctx:#?}");
    println!("{decls:#?}");
}
