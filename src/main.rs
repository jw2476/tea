use lex::tokenize;
use parse::parse_tokens;

mod lex;
mod parse;

fn main() {
    let file = String::from_utf8(std::fs::read("test.tea").unwrap()).unwrap();
    let tokens = tokenize(&file);
    let statements = parse_tokens(tokens);
    println!("{statements:?}");
}
