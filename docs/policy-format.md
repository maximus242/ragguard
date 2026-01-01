# Policy Format

Policies are simple YAML files that define who can access what.

## Basic Structure

```yaml
version: "1"

rules:
  # Rule 1: Public documents
  - name: "public-docs"
    match:
      visibility: "public"
    allow:
      everyone: true

  # Rule 2: Department-specific docs
  - name: "dept-docs"
    match:
      confidential: false
    allow:
      conditions:
        - user.department == document.department

  # Rule 3: Confidential docs (managers only, same dept)
  - name: "confidential-docs"
    match:
      confidential: true
    allow:
      roles: ["manager", "director"]
      conditions:
        - user.department == document.department

  # Rule 4: Shared documents (user in shared list)
  - name: "shared-docs"
    allow:
      conditions:
        - user.id in document.shared_with

  # Rule 5: Admin access to everything
  - name: "admin-access"
    allow:
      roles: ["admin"]

default: deny  # Deny by default if no rules match
```

## Supported Operators

### Equality (`==`)

Check if field equals a specific value:

```yaml
conditions:
  - user.department == document.department  # Same department
  - document.status == 'published'          # Literal value
```

### Inequality (`!=`)

Exclude documents with specific values:

```yaml
conditions:
  - document.access_level != 'restricted'   # Exclude restricted docs
  - document.status != 'archived'           # Exclude archived docs
  - user.id != document.author_id           # Exclude own documents
```

### List Membership (`in`)

Check if value is in a list:

```yaml
conditions:
  # Check if user value is in document's list field
  - user.id in document.shared_with

  # Check if document value is in a literal list
  - document.category in ['cs.AI', 'cs.LG', 'cs.CL']
  - document.region in ['us-east', 'us-west', 'eu-central']
```

### List Exclusion (`not in`)

Exclude documents with values in a list:

```yaml
conditions:
  # Exclude multiple statuses
  - document.status not in ['archived', 'deleted', 'draft']

  # Exclude specific categories
  - document.category not in ['restricted', 'classified']
```

### Comparison Operators (`>`, `<`, `>=`, `<=`)

```yaml
conditions:
  - document.security_level <= user.clearance_level
  - user.level >= 5
```

### Field Existence (`exists`, `not exists`)

```yaml
conditions:
  - document.approved_by exists
  - document.deleted_at not exists
```

## OR/AND Logic

Combine conditions with OR and AND for complex access patterns:

```yaml
rules:
  # Multiple roles: admin OR manager OR owner
  - name: "senior-access"
    allow:
      conditions:
        - "(user.role == 'admin' OR user.role == 'manager' OR user.role == 'owner')"

  # Shared documents: user in shared list OR document is public
  - name: "shared-or-public"
    allow:
      conditions:
        - "user.id in document.shared_with OR document.visibility == 'public'"

  # Complex nested: (admin OR manager) AND published
  - name: "senior-published-access"
    allow:
      conditions:
        - "(user.role == 'admin' OR user.role == 'manager') AND document.status == 'published'"
```

**Features:**
- Supports OR and AND operators (case-insensitive)
- Parentheses for grouping: `(A OR B) AND C`
- Unlimited nesting: `((A OR B) AND C) OR D`

## Nested Fields

Use dot notation for nested attributes:

```yaml
conditions:
  - user.team.id == document.owner.team_id
  - document.metadata.security.level != 'top-secret'
```

## Combining Operators

All operators can be combined with AND logic:

```yaml
rules:
  - name: "ai-papers-non-restricted"
    allow:
      everyone: true
      conditions:
        # Must be in allowed categories
        - document.category in ['cs.AI', 'cs.LG', 'cs.CV']
        # Must not be restricted
        - document.access_level != 'restricted'
        # Must not be in excluded statuses
        - document.status not in ['archived', 'deleted']
```
