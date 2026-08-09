def build_query(scope: str) -> str:
    if scope == "team-a":
        return "select * from evidence where acl_scope = 'team-a'"
    return "select * from evidence where acl_scope = 'internal'"
