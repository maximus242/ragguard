# Policy Syntax

RAGGuard policies define who can access which documents using a declarative JSON format.

## Basic Structure

```python
from ragguard import Policy

policy = Policy.from_dict({
    "version": "1",
    "rules": [
        {
            "name": "rule-name",
            "match": {"field": "value"},  # Optional: filter which documents this rule applies to
            "allow": {
                "everyone": True,           # OR
                "roles": ["admin", "user"], # OR
                "conditions": ["..."]       # Dynamic conditions
            }
        }
    ],
    "default": "deny"  # or "allow"
})
```

## Rule Evaluation

Rules are evaluated in order. The first matching rule determines access:

1. If `match` is specified, check if document matches
2. If match passes (or no match specified), evaluate `allow` conditions
3. If any allow condition passes, user can access the document
4. If no rules match, the `default` action applies

## Allow Conditions

### `everyone: True`
Allows all users to access matching documents.

```python
{"name": "public", "match": {"visibility": "public"}, "allow": {"everyone": True}}
```

### `roles: [...]`
Allows users with any of the specified roles.

```python
{"name": "admin", "allow": {"roles": ["admin", "superuser"]}}
```

### `conditions: [...]`
Dynamic conditions comparing user and document attributes.

```python
{"name": "dept", "allow": {"conditions": ["user.department == document.department"]}}
```

## Condition Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `==` | Equality | `user.id == document.owner` |
| `!=` | Inequality | `user.role != "guest"` |
| `<` | Less than | `document.level < user.clearance` |
| `<=` | Less than or equal | `document.min_level <= user.level` |
| `>` | Greater than | `user.clearance > document.classification` |
| `>=` | Greater than or equal | `user.level >= document.required_level` |
| `in` | Membership | `user.id in document.allowed_users` |
| `not in` | Non-membership | `user.id not in document.blocked_users` |

## Type Handling

**Important:** RAGGuard uses strict type comparison. Types must match exactly.

```python
# User context
user = {"id": "alice", "level": 5}       # level is integer

# Document metadata
document = {"required_level": "5"}        # required_level is STRING

# This condition will NOT match (int 5 != string "5")
"user.level >= document.required_level"
```

### Best Practices for Types

1. **Use consistent types** across user context and document metadata
2. **Prefer integers** for numeric comparisons
3. **Prefer strings** for identifiers and categories
4. **Document expected types** in your policy

```python
# Recommended: Both are integers
user = {"clearance": 3}
document = {"min_clearance": 2}
condition = "user.clearance >= document.min_clearance"  # Works!

# Not recommended: Mixed types
user = {"clearance": "3"}  # String
document = {"min_clearance": 2}  # Integer
# Condition will fail due to type mismatch
```

### Handling Missing Attributes

If a referenced attribute is missing or `None`, the condition evaluates to `False` (deny):

```python
user = {"id": "alice"}  # No "department" attribute
condition = "user.department == document.department"
# Result: DENY (missing attribute = no access)
```

This is a security feature: missing context defaults to deny.

## Match Filters

Match filters restrict which documents a rule applies to:

```python
{
    "name": "engineering-docs",
    "match": {"department": "engineering"},  # Only applies to engineering docs
    "allow": {"roles": ["engineer"]}
}
```

Match filters use exact equality. For lists, any value matches:

```python
{"match": {"category": ["tech", "science"]}}  # Matches tech OR science
```

## Complex Conditions

### Multiple Conditions (AND)

Multiple conditions in a single rule are combined with AND:

```python
{
    "name": "team-access",
    "allow": {
        "conditions": [
            "user.department == document.department",  # AND
            "user.level >= document.min_level"         # AND
        ]
    }
}
```

### OR Logic

Use separate rules for OR logic:

```python
{
    "rules": [
        {"name": "owner", "allow": {"conditions": ["user.id == document.owner"]}},
        {"name": "team", "allow": {"conditions": ["user.team == document.team"]}}
    ]
}
# User gets access if they're the owner OR on the team
```

## Limits and Validation

RAGGuard enforces limits to prevent DoS attacks:

| Limit | Default | Description |
|-------|---------|-------------|
| Max rules | 100 | Rules per policy |
| Max conditions per rule | 100 | Conditions in a single rule |
| Max total conditions | 1000 | Total across all rules |
| Max list size | 1000 | Elements in `in [...]` list |
| Max policy size | 1MB | Total policy JSON size |

## Examples

### Department-Based Access

```python
Policy.from_dict({
    "version": "1",
    "rules": [
        {
            "name": "same-department",
            "allow": {"conditions": ["user.department == document.department"]}
        }
    ],
    "default": "deny"
})
```

### Role + Clearance Level

```python
Policy.from_dict({
    "version": "1",
    "rules": [
        {"name": "admin", "allow": {"roles": ["admin"]}},
        {
            "name": "clearance",
            "allow": {"conditions": ["user.clearance >= document.classification"]}
        },
        {
            "name": "public",
            "match": {"visibility": "public"},
            "allow": {"everyone": True}
        }
    ],
    "default": "deny"
})
```

### Multi-Tenant with Sharing

```python
Policy.from_dict({
    "version": "1",
    "rules": [
        {"name": "owner", "allow": {"conditions": ["user.id == document.owner"]}},
        {"name": "shared", "allow": {"conditions": ["user.id in document.shared_with"]}},
        {"name": "org", "allow": {"conditions": ["user.org_id == document.org_id"]}}
    ],
    "default": "deny"
})
```
