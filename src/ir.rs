use crate::parse::{Context, Scope, ScopeKind};

pub fn prune_incomplete_blocks(ctx: &mut Context) {
    ctx.scopes
        .retain(|scope| !matches!(scope.kind, ScopeKind::IncompleteBlock))
}
