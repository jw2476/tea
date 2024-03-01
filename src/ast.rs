use std::{collections::HashMap, hash::Hash, ops::Index};

use crate::parse::{self, PrimitiveType, ProductType, Stat, SumType};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TypeId(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TypeClassId(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Resolved(TypeId),
    Unresolved(TypeClassId, Vec<TypeId>),
    Product(Vec<TypeId>),
    Sum(Vec<TypeId>),
    Function(TypeId, TypeId),
    Primitive(PrimitiveType),
    Generic(usize),
    Parsed {
        ty: parse::Type,
        generics: HashMap<String, TypeId>,
    },
}

#[derive(Clone, Debug)]
pub enum Node {
    Block,
    Get(NodeId, usize),
    Call(NodeId, NodeId),
    Product(Vec<NodeId>),
    Variant(usize, NodeId),
    Unwrap(NodeId, usize),
}

pub struct ScopedMap<K, V> {
    maps: Vec<HashMap<K, V>>,
}

impl<K: Eq + Hash, V> ScopedMap<K, V> {
    pub fn new() -> Self {
        Self { maps: Vec::new() }
    }

    pub fn push(&mut self) {
        self.maps.push(HashMap::new())
    }

    pub fn pop(&mut self) {
        self.maps.pop();
    }

    pub fn insert(&mut self, key: K, value: V) {
        self.maps.last_mut().unwrap().insert(key, value);
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.maps.iter().rev().find_map(|map| map.get(key))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Ast {
    types: Vec<Type>,
    nodes: Vec<Node>,
    next_generic: usize,
}

impl Ast {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_type(&mut self, ty: Type) -> TypeId {
        match self.types.iter().position(|t| t == &ty) {
            Some(index) => TypeId(index),
            None => {
                self.types.push(ty);
                TypeId(self.types.len() - 1)
            }
        }
    }

    pub fn add_generic(&mut self) -> TypeId {
        self.types.push(Type::Generic(self.next_generic));
        self.next_generic += 1;
        TypeId(self.types.len() - 1)
    }

    pub fn add_node(&mut self, node: Node) -> NodeId {
        self.nodes.push(node);
        NodeId(self.nodes.len() - 1)
    }
}

impl Index<TypeId> for Ast {
    type Output = Type;
    fn index(&self, index: TypeId) -> &Self::Output {
        &self.types[index.0]
    }
}

impl Index<NodeId> for Ast {
    type Output = Node;
    fn index(&self, index: NodeId) -> &Self::Output {
        &self.nodes[index.0]
    }
}

#[derive(Clone, Debug, Default)]
struct TypeClass {
    class: Vec<(Vec<TypeId>, TypeId)>,
}

impl TypeClass {
    pub fn push(&mut self, generics: Vec<TypeId>, ty: TypeId) {
        self.class.push((generics, ty))
    }

    pub fn get(&self, ast: &Ast, usage: Vec<TypeId>, types: Option<&TypeMap>) -> Option<TypeId> {
        self.class
            .iter()
            .filter(|(requirements, _)| requirements.len() == usage.len())
            .inspect(|(requirements, _)| println!("{:#?}", requirements))
            .map(|(requirements, x)| {
                if let Some(types) = types {
                    (
                        requirements
                            .iter()
                            .map(|requirement| match ast[*requirement].clone() {
                                Type::Unresolved(id, generics) => types.classes[id.0]
                                    .get(
                                        ast,
                                        generics
                                            .into_iter()
                                            .map(|generic| resolve(ast, generic, types))
                                            .collect(),
                                        Some(types),
                                    )
                                    .unwrap(),
                                x => *requirement,
                            })
                            .collect(),
                        x,
                    )
                } else {
                    (requirements.clone(), x)
                }
            })
            .filter(|(requirements, _)| {
                usage
                    .iter()
                    .zip(requirements)
                    .all(|x| match (&ast[*x.0], &ast[*x.1]) {
                        (Type::Unresolved(_, _), _) => panic!(),
                        (_, Type::Unresolved(id, generics)) => panic!(),
                        (_, Type::Generic(_)) => true,
                        (a, b) if a == b => true,
                        _ => false,
                    })
            })
            .map(|(requirements, ty)| {
                (
                    ty,
                    usage
                        .iter()
                        .zip(requirements)
                        .filter(|(a, b)| *a == b)
                        .count(),
                )
            })
            .max_by_key(|(_, score)| *score)
            .map(|(ty, _)| *ty)
    }
}

#[derive(Clone, Debug, Default)]
struct TypeMap {
    classes: Vec<TypeClass>,
    map: HashMap<String, TypeClassId>,
}

impl TypeMap {
    pub fn insert(&mut self, label: String, generics: Vec<TypeId>, ty: TypeId) {
        match self.map.get_mut(&label) {
            Some(types) => self.classes[types.0].push(generics, ty),
            None => {
                self.classes.push(TypeClass {
                    class: vec![(generics, ty)],
                });
                let tcid = TypeClassId(self.classes.len() - 1);
                self.map.insert(label, tcid);
            }
        }
    }

    pub fn get(
        &self,
        ast: &Ast,
        label: &str,
        usage: Vec<TypeId>,
        types: Option<&TypeMap>,
    ) -> Option<TypeId> {
        match self.map.get(label) {
            Some(tcid) => self.classes[tcid.0].get(ast, usage, types),
            None => None,
        }
    }
}

fn add_type(
    ast: &mut Ast,
    ty: parse::Type,
    types: &TypeMap,
    labels: &mut HashMap<TypeId, Vec<String>>,
    generics: &HashMap<String, TypeId>,
) -> TypeId {
    match ty {
        parse::Type::Sum(SumType { variants }) => {
            let inner = variants
                .iter()
                .map(|(_, ty)| add_type(ast, ty.clone(), types, labels, generics))
                .collect();
            let ty = ast.add_type(Type::Sum(inner));
            labels.insert(ty, variants.into_iter().map(|(label, _)| label).collect());
            ty
        }
        parse::Type::Product(ProductType { fields }) => {
            let inner = fields
                .iter()
                .map(|(_, ty)| add_type(ast, ty.clone(), types, labels, generics))
                .collect();
            let ty = ast.add_type(Type::Product(inner));
            labels.insert(ty, fields.into_iter().map(|(label, _)| label).collect());
            ty
        }
        parse::Type::Function(arg, ret) => {
            let arg = add_type(ast, *arg, types, labels, generics);
            let ret = add_type(ast, *ret, types, labels, generics);
            ast.add_type(Type::Function(arg, ret))
        }
        parse::Type::Named(label, gs) => {
            let generics = gs
                .into_iter()
                .map(|g| add_type(ast, g, types, labels, generics))
                .collect();
            let ty = Type::Resolved(
                types
                    .get(ast, &label, generics, Some(types))
                    .unwrap_or_else(|| panic!("Could not resolve {label}")),
            );
            ast.add_type(ty)
        }
        parse::Type::Generic(label) => *generics
            .get(&label)
            .unwrap_or_else(|| panic!("Undeclared generic {label} used")),
        parse::Type::Primitive(ty) => ast.add_type(Type::Primitive(ty)),
    }
}

fn to_generic(
    ast: &mut Ast,
    ty: parse::Type,
    types: &mut TypeMap,
    generics: &mut HashMap<String, TypeId>,
) -> TypeId {
    match ty {
        parse::Type::Generic(label) => {
            let ty = ast.add_generic();
            generics.insert(label, ty);
            ty
        }
        parse::Type::Named(label, gs) => {
            let generics = gs
                .into_iter()
                .map(|g| to_generic(ast, g, types, generics))
                .collect();

            let id = match types.map.get(&label) {
                Some(id) => *id,
                None => {
                    types.classes.push(TypeClass::default());
                    TypeClassId(types.classes.len() - 1)
                }
            };
            ast.add_type(Type::Unresolved(id, generics))
        }
        parse::Type::Primitive(ty) => ast.add_type(Type::Primitive(ty)),
        _ => todo!(),
    }
}

fn resolve(ast: &Ast, generic: TypeId, types: &TypeMap) -> TypeId {
    match ast[generic].clone() {
        Type::Unresolved(class, generics) => types.classes[class.0]
            .get(
                ast,
                generics
                    .into_iter()
                    .map(|generic| resolve(ast, generic, types))
                    .collect(),
                Some(types),
            )
            .unwrap(),
        _ => generic,
    }
}

pub fn to_ast(decls: Vec<Stat>) -> Ast {
    let mut ast = Ast::default();
    let mut types = TypeMap::default();

    // Add decls without resolving
    decls
        .iter()
        .filter_map(|decl| match decl {
            Stat::TDecl(label, generics, ty) => Some((label, generics, ty)),
            _ => None,
        })
        .for_each(|(label, generics, ty)| {
            let mut generic_labels = HashMap::new();
            let generics = generics
                .iter()
                .map(|generic| {
                    to_generic(&mut ast, generic.clone(), &mut types, &mut generic_labels)
                })
                .collect();

            types.insert(
                label.clone(),
                generics,
                ast.add_type(Type::Parsed {
                    ty: ty.clone(),
                    generics: generic_labels,
                }),
            );
        });

    // Resolve
    types.classes = types
        .classes
        .clone()
        .into_iter()
        .map(|class| TypeClass {
            class: class
                .class
                .into_iter()
                .map(|(generics, ty)| {
                    (
                        generics
                            .into_iter()
                            .map(|generic| resolve(&ast, generic, &types))
                            .collect(),
                        ty,
                    )
                })
                .collect(),
        })
        .collect();

    // Replace placeholders
    let mut labels = HashMap::new();
    types.classes.iter().for_each(|class| {
        class
            .class
            .iter()
            .for_each(|(_, id)| match ast[*id].clone() {
                Type::Parsed { ty, generics } => {
                    ast.types[id.0] =
                        Type::Resolved(add_type(&mut ast, ty, &types, &mut labels, &generics));
                }
                _ => panic!(),
            })
    });

    // Make all resolve chains direct
    loop {
        let old = ast.types.clone();
        ast.types = old
            .clone()
            .into_iter()
            .map(|ty| match ty {
                Type::Resolved(first) => match ast[first] {
                    Type::Resolved(second) => Type::Resolved(second),
                    _ => Type::Resolved(first),
                },
                x => x,
            })
            .collect();
        if ast.types == old {
            break;
        }
    }

    println!("{:#?}", types);
    println!("{:#?}", ast);

    ast
}
