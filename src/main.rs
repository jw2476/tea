use ir::generate;
use lex::tokenize;
use parse::parse_tokens;

mod ir;
mod lex;
mod parse;

fn main() {
    let file = String::from_utf8(std::fs::read("test.tea").unwrap()).unwrap();
    let tokens = tokenize(&file);
    let (ctx, decls) = parse_tokens(&tokens).unwrap();
    let mut ir = generate(&ctx, decls);
    println!("{}", ir.display());
    ir.resolve();

    println!("{}", ir.display());
}
