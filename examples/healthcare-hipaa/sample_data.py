"""
Generate and load sample medical records into ChromaDB.

This script creates realistic (but fictional) medical records for testing
the HIPAA-compliant Q&A system.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
import random
import json

# Import RAGGuard (optional, just for path setup)
import sys
sys.path.insert(0, '../../')

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Delete existing collection if it exists
try:
    chroma_client.delete_collection("medical_records")
    print("Deleted existing collection")
except:
    pass

collection = chroma_client.create_collection(
    name="medical_records",
    metadata={"description": "HIPAA-compliant medical records"}
)

# Sample medical records
medical_records = [
    # Patient P001 - John Smith (Cardiology)
    {
        "text": "Patient John Smith, age 65, diagnosed with hypertension. Blood pressure reading: 145/90. Prescribed Lisinopril 10mg daily. Follow-up in 2 weeks.",
        "metadata": {
            "patient_id": "P001",
            "patient_name": "John Smith",
            "department": "cardiology",
            "assigned_doctor": "dr.williams@hospital.com",
            "record_type": "visit_note",
            "date": "2024-01-10",
            "deidentified": False
        }
    },
    {
        "text": "John Smith - Lab results: Cholesterol 240 mg/dL (high), LDL 160 mg/dL, HDL 45 mg/dL. Recommend starting statin therapy and dietary modifications.",
        "metadata": {
            "patient_id": "P001",
            "patient_name": "John Smith",
            "department": "cardiology",
            "assigned_doctor": "dr.williams@hospital.com",
            "record_type": "lab_result",
            "date": "2024-01-12",
            "deidentified": False
        }
    },
    {
        "text": "Patient P001 (John Smith) - Type 2 Diabetes diagnosis confirmed. HbA1c: 7.8%. Started on Metformin 500mg twice daily. Diabetes education scheduled.",
        "metadata": {
            "patient_id": "P001",
            "patient_name": "John Smith",
            "department": "cardiology",
            "assigned_doctor": "dr.williams@hospital.com",
            "record_type": "diagnosis",
            "date": "2024-01-15",
            "deidentified": False
        }
    },
    {
        "text": "Follow-up visit for John Smith. Blood pressure improved to 130/85 with medication. Continue current regimen. Blood glucose fasting: 125 mg/dL.",
        "metadata": {
            "patient_id": "P001",
            "patient_name": "John Smith",
            "department": "cardiology",
            "assigned_doctor": "dr.williams@hospital.com",
            "record_type": "visit_note",
            "date": "2024-01-20",
            "deidentified": False
        }
    },

    # Patient P002 - Jane Doe (Pulmonology)
    {
        "text": "Patient Jane Doe, age 42, presenting with persistent cough and shortness of breath. Diagnosed with moderate persistent asthma. Prescribed Albuterol inhaler.",
        "metadata": {
            "patient_id": "P002",
            "patient_name": "Jane Doe",
            "department": "pulmonology",
            "assigned_doctor": "dr.chen@hospital.com",
            "record_type": "visit_note",
            "date": "2024-01-08",
            "deidentified": False
        }
    },
    {
        "text": "Jane Doe - Pulmonary function test results: FEV1 75% predicted, FVC 82% predicted. Consistent with moderate asthma. Started on inhaled corticosteroid.",
        "metadata": {
            "patient_id": "P002",
            "patient_name": "Jane Doe",
            "department": "pulmonology",
            "assigned_doctor": "dr.chen@hospital.com",
            "record_type": "test_result",
            "date": "2024-01-11",
            "deidentified": False
        }
    },
    {
        "text": "Follow-up for Jane Doe: Asthma symptoms well-controlled with current medication regimen. Peak flow measurements improved. Continue current treatment.",
        "metadata": {
            "patient_id": "P002",
            "patient_name": "Jane Doe",
            "department": "pulmonology",
            "assigned_doctor": "dr.chen@hospital.com",
            "record_type": "visit_note",
            "date": "2024-01-18",
            "deidentified": False
        }
    },

    # Patient P003 - Bob Johnson (Orthopedics)
    {
        "text": "Patient Bob Johnson, age 35, sports injury - torn ACL in right knee. X-ray shows no fracture. MRI recommended. Referred to orthopedic surgery for consultation.",
        "metadata": {
            "patient_id": "P003",
            "patient_name": "Bob Johnson",
            "department": "orthopedics",
            "assigned_doctor": "dr.martinez@hospital.com",
            "record_type": "visit_note",
            "date": "2024-01-05",
            "deidentified": False
        }
    },
    {
        "text": "Bob Johnson - MRI confirms complete ACL tear. Surgery recommended. Patient consents to arthroscopic ACL reconstruction. Pre-op clearance obtained.",
        "metadata": {
            "patient_id": "P003",
            "patient_name": "Bob Johnson",
            "department": "orthopedics",
            "assigned_doctor": "dr.martinez@hospital.com",
            "record_type": "imaging_report",
            "date": "2024-01-09",
            "deidentified": False
        }
    },
    {
        "text": "Surgical note - Patient P003 (Bob Johnson): Successful arthroscopic ACL reconstruction. Patellar tendon autograft used. No complications. Post-op recovery normal.",
        "metadata": {
            "patient_id": "P003",
            "patient_name": "Bob Johnson",
            "department": "orthopedics",
            "assigned_doctor": "dr.martinez@hospital.com",
            "record_type": "surgical_note",
            "date": "2024-01-16",
            "deidentified": False
        }
    },

    # De-identified records for research (all patients can access this)
    {
        "text": "De-identified study data: 65-year-old male with hypertension and diabetes. Treatment with ACE inhibitor and metformin showed good outcomes. Blood pressure normalized within 4 weeks.",
        "metadata": {
            "patient_id": "DEIDENTIFIED_001",
            "patient_name": "De-identified",
            "department": "research",
            "assigned_doctor": "",
            "record_type": "research_data",
            "date": "2024-01-01",
            "deidentified": True
        }
    },
    {
        "text": "Research cohort: 42-year-old female with moderate asthma. Combination therapy (ICS + LABA) resulted in 80% symptom reduction over 12 weeks.",
        "metadata": {
            "patient_id": "DEIDENTIFIED_002",
            "patient_name": "De-identified",
            "department": "research",
            "assigned_doctor": "",
            "record_type": "research_data",
            "date": "2024-01-01",
            "deidentified": True
        }
    },

    # Billing records (accessible to billing staff)
    {
        "text": "Billing record for patient P001: Office visit (99214), Lab tests (80053, 83036), Total charges: $425.00. Insurance: Medicare. Co-pay collected: $25.00.",
        "metadata": {
            "patient_id": "P001",
            "patient_name": "John Smith",
            "department": "billing",
            "assigned_doctor": "",
            "record_type": "billing",
            "date": "2024-01-12",
            "deidentified": False
        }
    },
]

print(f"\nLoading {len(medical_records)} medical records...")

# Add records to collection
ids = []
documents = []
embeddings = []
metadatas = []

for i, record in enumerate(medical_records):
    record_id = f"record_{i+1}"
    ids.append(record_id)
    documents.append(record["text"])

    # ChromaDB metadata - no conversion needed now
    metadatas.append(record["metadata"])

    # Generate embedding
    embedding = embedder.encode(record["text"]).tolist()
    embeddings.append(embedding)

    print(f"  [{i+1}/{len(medical_records)}] {record['metadata']['patient_id']} - {record['metadata']['record_type']}")

# Bulk add to ChromaDB
collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print(f"\n✓ Successfully loaded {len(medical_records)} records into ChromaDB")
print(f"\nCollection stats:")
print(f"  Total documents: {collection.count()}")
print(f"  Patients: P001 (John Smith), P002 (Jane Doe), P003 (Bob Johnson)")
print(f"  Departments: Cardiology, Pulmonology, Orthopedics")
print(f"\nSample users:")
print(f"  Patient: john.smith@email.com (role: patient, patient_id: P001)")
print(f"  Doctor: dr.williams@hospital.com (role: doctor, assigned_patients: [P001])")
print(f"  Nurse: nurse.davis@hospital.com (role: nurse, department: cardiology)")
print(f"  Admin: admin.brown@hospital.com (role: admin)")
print(f"\nRun 'python app.py' to start the Q&A system!")
