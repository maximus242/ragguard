#!/bin/bash

# Test script for Healthcare HIPAA Q&A System
# This script demonstrates different access control scenarios

API_URL="http://localhost:8000"

echo "=================================="
echo "Healthcare HIPAA Q&A System Tests"
echo "=================================="
echo ""

# Check if API is running
echo "1. Checking if API is running..."
curl -s "$API_URL/" | grep -q "healthy" && echo "✓ API is healthy" || echo "✗ API is not running"
echo ""

# Test 1: Patient accessing own records
echo "2. Test: Patient accessing own records"
echo "   User: john.smith@email.com (Patient P001)"
curl -s -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is my blood pressure?",
    "user": {
      "id": "john.smith@email.com",
      "roles": ["patient"],
      "patient_id": "P001"
    },
    "limit": 3
  }' | python3 -m json.tool
echo ""

# Test 2: Doctor accessing assigned patient
echo "3. Test: Doctor accessing assigned patient"
echo "   User: dr.williams@hospital.com (Doctor assigned to P001)"
curl -s -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show recent visits for my patients",
    "user": {
      "id": "dr.williams@hospital.com",
      "roles": ["doctor"],
      "department": "cardiology",
      "assigned_patients": ["P001"]
    },
    "limit": 3
  }' | python3 -m json.tool
echo ""

# Test 3: Nurse accessing department patients
echo "4. Test: Nurse accessing department patients"
echo "   User: nurse.davis@hospital.com (Cardiology)"
curl -s -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show cardiology patient medications",
    "user": {
      "id": "nurse.davis@hospital.com",
      "roles": ["nurse"],
      "department": "cardiology"
    },
    "limit": 3
  }' | python3 -m json.tool
echo ""

# Test 4: Admin full access
echo "5. Test: Admin full access"
echo "   User: admin.brown@hospital.com (Admin)"
curl -s -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show all patient records",
    "user": {
      "id": "admin.brown@hospital.com",
      "roles": ["admin"]
    },
    "limit": 5
  }' | python3 -m json.tool
echo ""

# Test 5: Unauthorized access (patient trying to access another patient's records)
echo "6. Test: Unauthorized access attempt"
echo "   User: jane.doe@email.com trying to access P001 records"
curl -s -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show John Smith medical records",
    "user": {
      "id": "jane.doe@email.com",
      "roles": ["patient"],
      "patient_id": "P002"
    },
    "limit": 3
  }' | python3 -m json.tool
echo ""

# Show audit logs
echo "7. Audit Logs (last 5 entries)"
curl -s "$API_URL/audit/recent?limit=5" | python3 -m json.tool
echo ""

echo "=================================="
echo "Tests completed!"
echo "=================================="
