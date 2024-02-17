struct Vec2(u32, u32);

enum Option {
    Some(u32),
    None,
}

fn unwrap(x: Option) -> u32 {
    match x {
        Some(x) => x,
        None => panic!(),
    }
}

fn add(a: u32, b: u32) -> u32 {
    a
}

fn add(a: Vec2, b: Vec2) -> Vec2 {
    Vec2(add(a.0, b.0), add(a.1, b.1))
}

fn map<F: Fn(u32) -> u32>(x: Option, f: F) -> Option {
    match x {
        Some(x) => Some(f(x)),
        None => None,
    }
}
