# Use Cases

## Multi-Tenant SaaS

Isolate data between customers:

```yaml
rules:
  - name: "tenant-isolation"
    allow:
      conditions:
        - user.tenant_id == document.tenant_id
default: deny
```

## Healthcare (HIPAA)

Doctors see their patients, patients see their own records:

```yaml
rules:
  - name: "doctor-access"
    allow:
      roles: ["doctor"]
      conditions:
        - user.id in document.assigned_doctors

  - name: "patient-access"
    allow:
      roles: ["patient"]
      conditions:
        - user.id == document.patient_id
default: deny
```

## Enterprise Knowledge Base

Department-based access with cross-functional sharing:

```yaml
rules:
  - name: "department-docs"
    allow:
      conditions:
        - user.department == document.department

  - name: "shared-docs"
    allow:
      conditions:
        - user.email in document.shared_with

  - name: "company-wide"
    match:
      scope: "company"
    allow:
      everyone: true
default: deny
```

## Legal/Finance

Attorney-client privilege and compliance:

```yaml
rules:
  - name: "attorney-client"
    match:
      privileged: true
    allow:
      roles: ["attorney"]
      conditions:
        - user.bar_number in document.authorized_attorneys

  - name: "public-filings"
    match:
      public_filing: true
    allow:
      everyone: true
default: deny
```

## Research Institution

Academic access with publication embargoes:

```yaml
rules:
  - name: "published-papers"
    match:
      status: "published"
    allow:
      everyone: true

  - name: "lab-members"
    allow:
      conditions:
        - user.lab_id == document.lab_id

  - name: "collaborators"
    allow:
      conditions:
        - user.id in document.collaborators
default: deny
```

## Government/Classified

Clearance-based access:

```yaml
rules:
  - name: "clearance-check"
    allow:
      conditions:
        - user.clearance_level >= document.classification_level

  - name: "need-to-know"
    allow:
      conditions:
        - user.project_id == document.project_id
default: deny
```
