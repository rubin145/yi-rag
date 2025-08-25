#!/usr/bin/env python3
"""
Script to fix indexes for the test collection yi_rag_test
"""

from core.store_qdrant import get_client
from qdrant_client import models as qm

def fix_test_indexes():
    client = get_client()
    collection = "yi_rag_test"
    
    print(f"Fixing indexes for collection: {collection}")
    
    # Create keyword indexes for filter fields
    keyword_fields = ["author", "author_yi", "place", "publisher"]
    
    for field in keyword_fields:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=qm.PayloadSchemaType.KEYWORD
            )
            print(f"✅ Created KEYWORD index for {field}")
        except Exception as e:
            print(f"❌ Failed to create index for {field}: {e}")
    
    # Verify indexes exist
    try:
        collection_info = client.get_collection(collection)
        print(f"\n📋 Collection info:")
        print(f"Points count: {collection_info.points_count}")
        print(f"Vectors count: {collection_info.vectors_count}")
        
        # Try to list payload indexes if available
        try:
            indexes = client.list_indexes(collection_name=collection)
            print(f"Existing indexes: {indexes}")
        except:
            print("Could not list indexes (might not be supported in this Qdrant version)")
            
    except Exception as e:
        print(f"❌ Could not get collection info: {e}")

if __name__ == "__main__":
    fix_test_indexes()