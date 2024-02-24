#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    Star,
    Underscore,
    Equals,
    Semicolon,
    Colon,
    DoubleColon,
    Tick,
    Pipe,
    Comma,
    Dot,
    Arrow,
    WideArrow,
    Backslash,
    Bang,

    LeftRound,
    RightRound,
    LeftSquare,
    RightSquare,
    LeftCurly,
    RightCurly,
    LeftAngle,
    RightAngle,

    Identifier(String),
    Integer(String),
    Decimal(String),
    String(String),
}

pub fn tokenize(text: &str) -> Vec<Token> {
    let mut chars = text.chars().peekable();
    let mut tokens = Vec::new();
    loop {
        let Some(c) = chars.next() else {
            break;
        };
        let token = match c {
            '!' => Token::Bang,
            '*' => Token::Star,
            '_' => Token::Underscore,
            '=' => {
                if *chars.peek().unwrap() == '>' {
                    chars.next();
                    Token::WideArrow
                } else {
                    Token::Equals
                }
            }
            ';' => Token::Semicolon,
            ':' => {
                if *chars.peek().unwrap() == ':' {
                    chars.next();
                    Token::DoubleColon
                } else {
                    Token::Colon
                }
            }
            '\'' => Token::Tick,
            '|' => Token::Pipe,
            ',' => Token::Comma,
            '.' => Token::Dot,
            '-' => {
                if *chars.peek().unwrap() == '>' {
                    chars.next();
                    Token::Arrow
                } else {
                    panic!("Should be arrow")
                }
            }
            '\\' => Token::Backslash,

            '(' => Token::LeftRound,
            ')' => Token::RightRound,
            '[' => Token::LeftSquare,
            ']' => Token::RightSquare,
            '{' => Token::LeftCurly,
            '}' => Token::RightCurly,
            '<' => Token::LeftAngle,
            '>' => Token::RightAngle,

            '/' => {
                while *chars.peek().unwrap() != '\n' {
                    chars.next();
                }
                continue;
            }

            c if c.is_alphabetic() => {
                let mut data = c.to_string();
                while chars.peek().unwrap().is_alphanumeric() || *chars.peek().unwrap() == '_' {
                    data += &chars.next().unwrap().to_string()
                }
                Token::Identifier(data)
            }
            c if c.is_ascii_digit() => {
                let mut data = c.to_string();
                let mut decimal = false;

                loop {
                    if *chars.peek().unwrap() == '.' && chars.clone().nth(1).unwrap() == '.' {
                        decimal = true;
                    } else if !chars.peek().unwrap().is_numeric() {
                        break;
                    }
                    data += &chars.next().unwrap().to_string();
                }

                if decimal {
                    Token::Decimal(data)
                } else {
                    Token::Integer(data)
                }
            }
            '"' => {
                let mut data = String::new();
                while *chars.peek().unwrap() != '"' {
                    data += &chars.next().unwrap().to_string();
                }
                chars.next();
                Token::String(data)
            }
            ' ' | '\n' | '\t' => continue,
            _ => panic!("Unknown char {}", c),
        };
        tokens.push(token);
    }
    tokens
}
