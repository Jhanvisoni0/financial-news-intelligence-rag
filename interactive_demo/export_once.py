# interactive_demo/export_once.py
# Run this ONCE while your Databricks cluster is running, using the same
# credentials as app_live.py. It saves your real Gold table to a local file
# (gold_chunks_cache.parquet), so the offline app never needs Databricks again.
#
# Run with: python export_once.py

import getpass
import pandas as pd
from databricks import sql as databricks_sql

print("=== One-time export of real Gold data from Databricks ===\n")

hostname = input("Server Hostname [adb-7405612624543070.10.azuredatabricks.net]: ") or "adb-7405612624543070.10.azuredatabricks.net"
http_path = input("HTTP Path [sql/protocolv1/o/7405612624543070/0703-031906-eib1v6oy]: ") or "sql/protocolv1/o/7405612624543070/0703-031906-eib1v6oy"
token = getpass.getpass("Databricks Access Token (hidden as you type): ")

print("\nConnecting and querying gold.embedded_chunks...")

connection = databricks_sql.connect(
    server_hostname=hostname,
    http_path=http_path,
    access_token=token,
)
cursor = connection.cursor()
cursor.execute("""
    SELECT chunk_id, ticker, source_type, doc_subtype, date,
           chunk_text, sentiment, risk_density, embedding
    FROM gold.embedded_chunks
""")
rows = cursor.fetchall()
columns = [desc[0] for desc in cursor.description]
cursor.close()
connection.close()

df = pd.DataFrame(rows, columns=columns)
print(f"Retrieved {len(df)} chunks.")

output_file = "gold_chunks_cache.parquet"
df.to_parquet(output_file, index=False)
print(f"\nSaved to: {output_file}")
print("You can now run 'streamlit run app_offline.py' anytime, with no Databricks/Azure connection needed.")
