use crate::lex::Token;

pub struct Tokens {
    index: usize,
    tokens: Vec<Token>,
}

impl Tokens {
    pub fn peek(&self) -> Token {
        self.tokens[self.index].clone()
    }

    pub fn next(&mut self) -> Token {
        self.index += 1;
        self.tokens[self.index - 1].clone()
    }
}

trait Parser: Sized {
    fn parse(tokens: &mut Tokens) -> Option<Self>;
}

fn parse<T: Parser>(tokens: &mut Tokens) -> Option<T> {
    let start = tokens.index;
    let option = T::parse(tokens);
    if option.is_none() {
        tokens.index = start;
    }
    option
}

#[derive(Debug, Clone)]
pub struct Sum {
    variants: Vec<(String, Expression)>,
}

impl Parser for Sum {
    fn parse(tokens: &mut Tokens) -> Option<Self> {
        let Token::LeftSquare = tokens.next() else {
            return None;
        };
        let Token::Identifier(ident) = tokens.next() else {
            return None;
        };
        let expr = if let Token::Comma = tokens.peek() {
            Expression {
                parts: vec![Part::Product(Product::empty())],
            }
        } else {
            parse(tokens).unwrap()
        };
        let mut variants = vec![(ident, expr)];
        loop {
            match tokens.peek() {
                Token::RightSquare => {
                    tokens.next();
                    return Some(Self { variants });
                }
                Token::Comma => {
                    tokens.next();
                    let Token::Identifier(ident) = tokens.next() else {
                        return None;
                    };
                    let expr = match tokens.peek() {
                        Token::Comma | Token::RightSquare => Expression {
                            parts: vec![Part::Product(Product::empty())],
                        },
                        _ => parse(tokens).unwrap(),
                    };
                    variants.push((ident, expr))
                }
                _ => panic!("Invalid token in sum type"),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Product {
    fields: Vec<Expression>,
    labels: Vec<String>,
}

impl Parser for Product {
    fn parse(tokens: &mut Tokens) -> Option<Self> {
        //println!("{:?}", &tokens.tokens[tokens.index..tokens.index + 10]);
        let Token::LeftCurly = tokens.next() else {
            return None;
        };
        if let Token::RightCurly = tokens.peek() {
            tokens.next();
            return Some(Self {
                fields: Vec::new(),
                labels: Vec::new(),
            });
        };
        let start = tokens.index;
        let labelled = if parse::<Expression>(tokens).is_some() {
            !matches!(tokens.peek(), Token::Comma)
        } else {
            true
        };
        tokens.index = start;

        let (field, label) = if labelled {
            let Token::Identifier(label) = tokens.next() else {
                panic!()
            };
            (parse(tokens)?, Some(label))
        } else {
            (parse(tokens)?, None)
        };
        let mut fields = vec![field];
        let mut labels = label.map(|x| vec![x]).unwrap_or_default();

        println!("{fields:?}");

        loop {
            match tokens.peek() {
                Token::RightCurly => {
                    tokens.next();
                    return Some(Self { fields, labels });
                }
                Token::Comma => {
                    tokens.next();
                    if labelled {
                        let Token::Identifier(label) = tokens.next() else {
                            return None;
                        };
                        labels.push(label)
                    }
                    fields.push(parse(tokens)?);
                }
                _ => panic!("Invalid token in product type"),
            }
        }
    }
}

impl Product {
    pub fn empty() -> Self {
        Self {
            fields: Vec::new(),
            labels: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Argument {
    Sum(Sum),
    Product(Product),
    Call(Call),
    Hole(Hole),
    Variable(String),
    Other(Expression),
}

impl Parser for Argument {
    fn parse(tokens: &mut Tokens) -> Option<Self> {
        if let Some(part) = parse(tokens) {
            Some(Self::Sum(part))
        } else if let Some(part) = parse(tokens) {
            Some(Self::Product(part))
        } else if let Some(part) = parse(tokens) {
            Some(Self::Call(part))
        } else if let Some(part) = parse(tokens) {
            Some(Self::Hole(part))
        } else {
            Some(match tokens.next() {
                Token::LeftRound => {
                    let other = Self::Other(parse(tokens)?);
                    tokens.next();
                    other
                }
                Token::Identifier(id) => Self::Variable(id),
                _ => return None,
            })
        }
    }
}

#[derive(Debug, Clone)]
pub struct Function {
    argument: Argument,
    return_type: Expression,
}

impl Parser for Function {
    fn parse(tokens: &mut Tokens) -> Option<Self> {
        let argument = parse(tokens)?;
        let Token::Arrow = tokens.next() else {
            return None;
        };
        let return_type = parse(tokens)?;
        Some(Self {
            argument,
            return_type,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Call {
    function: String,
    arguments: Vec<Expression>,
}

impl Parser for Call {
    fn parse(tokens: &mut Tokens) -> Option<Self> {
        let Token::Identifier(function) = tokens.next() else {
            return None;
        };
        let Token::LeftRound = tokens.next() else {
            return None;
        };
        if let Token::RightRound = tokens.peek() {
            tokens.next();
            return Some(Self {
                function,
                arguments: Vec::new(),
            });
        }
        let mut arguments = vec![parse(tokens)?];
        loop {
            match tokens.peek() {
                Token::RightRound => {
                    tokens.next();
                    return Some(Self {
                        function,
                        arguments,
                    });
                }
                Token::Comma => {
                    tokens.next();
                    arguments.push(parse(tokens)?)
                }
                _ => panic!("Unknown token in call arguments"),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum PathPart {
    Call(Call),
    Hole(Hole),
    Variable(String),
    String(String),
    Decimal(String),
    Integer(String),
}

impl Parser for PathPart {
    fn parse(tokens: &mut Tokens) -> Option<Self> {
        if let Some(part) = parse(tokens) {
            Some(Self::Call(part))
        } else if let Some(part) = parse(tokens) {
            Some(Self::Hole(part))
        } else {
            Some(match tokens.next() {
                Token::Identifier(id) => Self::Variable(id),
                Token::String(string) => Self::String(string),
                Token::Decimal(decimal) => Self::Decimal(decimal),
                Token::Integer(integer) => Self::Integer(integer),
                _ => return None,
            })
        }
    }
}

#[derive(Debug, Clone)]
pub struct Path {
    parts: Vec<PathPart>,
}

impl Parser for Path {
    fn parse(tokens: &mut Tokens) -> Option<Self> {
        let mut parts = vec![parse(tokens)?];
        loop {
            if let Token::Dot = tokens.peek() {
                tokens.next();
                parts.push(parse(tokens)?)
            } else {
                return Some(Self { parts });
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Variant {
    path: Path,
    value: Expression,
}

impl Parser for Variant {
    fn parse(tokens: &mut Tokens) -> Option<Self> {
        let path = parse(tokens)?;
        let value = parse(tokens)?;
        Some(Self { path, value })
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    statements: Vec<Statement>,
    expression: Option<Expression>,
}

impl Parser for Block {
    fn parse(tokens: &mut Tokens) -> Option<Self> {
        let Token::LeftCurly = tokens.next() else {
            return None;
        };
        let mut statements = Vec::new();
        while let Some(statement) = parse(tokens) {
            statements.push(statement)
        }
        if let Token::RightCurly = tokens.peek() {
            tokens.next();
            Some(Self {
                statements,
                expression: None,
            })
        } else {
            Some(Self {
                statements,
                expression: Some(parse(tokens)?),
            })
        }
    }
}

#[derive(Debug, Clone)]
pub struct Lambda {
    arguments: Option<String>,
    body: Expression,
}

impl Parser for Lambda {
    fn parse(tokens: &mut Tokens) -> Option<Self> {
        let Token::Pipe = tokens.next() else {
            return None;
        };
        let arguments = if let Token::Pipe = tokens.peek() {
            None
        } else {
            match tokens.next() {
                Token::Identifier(ident) => Some(ident),
                Token::Underscore => None,
                _ => return None,
            }
        };
        println!("{:?}", &tokens.tokens[tokens.index..tokens.index + 10]);
        tokens.next();
        let body = parse(tokens)?;
        Some(Self { arguments, body })
    }
}

#[derive(Debug, Clone)]
pub struct Hole {
    id: Option<String>,
}

impl Parser for Hole {
    fn parse(tokens: &mut Tokens) -> Option<Self> {
        let Token::Tick = tokens.next() else {
            return None;
        };
        let id = if let Token::Underscore = tokens.peek() {
            tokens.next();
            None
        } else if let Token::Identifier(id) = tokens.peek() {
            tokens.next();
            Some(id)
        } else {
            panic!("Invalid token in type hole")
        };
        Some(Self { id })
    }
}

#[derive(Debug, Clone)]
pub enum Part {
    Lambda(Lambda),
    Function(Function),
    Sum(Sum),
    Product(Product),
    Block(Block),
    Call(Call),
    Variant(Variant),
    Hole(Hole),
    Variable(Option<String>),
    String(String),
    Decimal(String),
    Integer(String),
}

impl Parser for Part {
    fn parse(tokens: &mut Tokens) -> Option<Self> {
        if let Some(part) = parse(tokens) {
            Some(Self::Lambda(part))
        } else if let Some(part) = parse(tokens) {
            Some(Self::Function(part))
        } else if let Some(part) = parse(tokens) {
            Some(Self::Sum(part))
        } else if let Some(part) = parse(tokens) {
            Some(Self::Product(part))
        } else if let Some(part) = parse(tokens) {
            Some(Self::Block(part))
        } else if let Some(part) = parse(tokens) {
            Some(Self::Call(part))
        } else if let Some(part) = parse(tokens) {
            Some(Self::Variant(part))
        } else if let Some(part) = parse(tokens) {
            Some(Self::Hole(part))
        } else {
            Some(match tokens.next() {
                Token::Identifier(id) => Self::Variable(Some(id)),
                Token::Underscore => Self::Variable(None),
                Token::String(string) => Self::String(string),
                Token::Decimal(decimal) => Self::Decimal(decimal),
                Token::Integer(integer) => Self::Integer(integer),
                _ => return None,
            })
        }
    }
}

#[derive(Debug, Clone)]
pub struct Expression {
    parts: Vec<Part>,
}

impl Parser for Expression {
    fn parse(tokens: &mut Tokens) -> Option<Self> {
        let mut parts = vec![parse(tokens)?];
        loop {
            if let Token::Dot = tokens.peek() {
                tokens.next();
                parts.push(parse(tokens)?)
            } else {
                return Some(Self { parts });
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TypeDecl {
    ident: String,
    ty: Expression,
}

impl Parser for TypeDecl {
    fn parse(tokens: &mut Tokens) -> Option<Self> {
        let Token::Identifier(ident) = tokens.next() else {
            return None;
        };
        let Token::DoubleColon = tokens.next() else {
            return None;
        };
        let ty = parse(tokens)?;
        println!("{ty:?}");
        let Token::Semicolon = tokens.next() else {
            return None;
        };
        Some(Self { ident, ty })
    }
}

#[derive(Debug, Clone)]
pub struct ValueDecl {
    ident: String,
    pattern: Option<Expression>,
    value: Expression,
}

impl Parser for ValueDecl {
    fn parse(tokens: &mut Tokens) -> Option<Self> {
        let Token::Identifier(ident) = tokens.next() else {
            return None;
        };
        let pattern = if let Token::Equals = tokens.peek() {
            None
        } else {
            Some(parse(tokens)?)
        };
        tokens.next(); // Skip =
        let value = parse(tokens)?;
        let Token::Semicolon = tokens.next() else {
            return None;
        };
        Some(Self {
            ident,
            pattern,
            value,
        })
    }
}

#[derive(Debug, Clone)]
pub struct CombinedDecl {
    ident: String,
    ty: Expression,
    value: Expression,
}

impl Parser for CombinedDecl {
    fn parse(tokens: &mut Tokens) -> Option<Self> {
        let Token::Identifier(ident) = tokens.next() else {
            return None;
        };
        let Token::DoubleColon = tokens.next() else {
            return None;
        };
        let ty = parse(tokens)?;
        let Token::Equals = tokens.next() else {
            return None;
        };
        let value = parse(tokens)?;
        let Token::Semicolon = tokens.next() else {
            return None;
        };
        Some(Self { ident, ty, value })
    }
}

#[derive(Debug, Clone)]
pub enum Statement {
    Type(TypeDecl),
    Value(ValueDecl),
    Combined(CombinedDecl),
}

impl Parser for Statement {
    fn parse(tokens: &mut Tokens) -> Option<Self> {
        if let Some(inner) = parse(tokens) {
            Some(Self::Type(inner))
        } else if let Some(inner) = parse(tokens) {
            Some(Self::Value(inner))
        } else if let Some(inner) = parse(tokens) {
            Some(Self::Combined(inner))
        } else {
            None
        }
    }
}

pub fn parse_tokens(tokens: Vec<Token>) -> Vec<Statement> {
    let mut tokens = Tokens { index: 0, tokens };
    let mut statements = Vec::new();
    while tokens.index != tokens.tokens.len() {
        statements.push(parse(&mut tokens).unwrap());
        println!("{:?}", statements.last().unwrap());
    }
    statements
}
