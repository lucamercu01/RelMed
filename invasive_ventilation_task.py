from relbench.base import TaskType, EntityTask, Database, Table
from relbench.metrics import accuracy, average_precision, f1, roc_auc, auprc
import duckdb
import pandas as pd
import numpy as np

class CXRVentilationTask(EntityTask):
    r"""Binary Classification: Predict if a patient will require Invasive Ventilation
    within 24 hours of the Chest X-Ray."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "subject_id"
    entity_table = "patients"
    time_col = "study_time"
    target_col = "ventilation_24h"
    timedelta = pd.Timedelta(hours=24)
    metrics = [roc_auc, auprc, accuracy, f1]

    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.test_timestamp = dataset.test_timestamp
        self.val_timestamp = dataset.val_timestamp

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        # 1. Load tables
        metadata = db.table_dict["metadata"].df
        procedureevents = db.table_dict["procedureevents"].df
        d_items = db.table_dict["d_items"].df

        # 2. Dynamic ID Lookup (Crucial Step!)
        # We find the internal itemid for "Invasive Ventilation"
        # We search for labels containing "Invasive Ventilation" or "Mechanical Ventilation"
        target_labels = ["Invasive Ventilation", "Mechanical Ventilation"]

        # Get the list of IDs that match these labels
        vent_ids_list = d_items[d_items['label'].isin(target_labels)]['itemid'].tolist()

        if not vent_ids_list:
            print("WARNING: No ventilation labels found in d_items. Task will be empty.")
            vent_ids_sql = "(-1)" # Dummy ID to prevent crash
        else:
            # Format as SQL tuple: "(274, 105, ...)"
            vent_ids_sql = f"({', '.join(map(str, vent_ids_list))})"
            print(f"Found Ventilation IDs: {vent_ids_sql}")

        # 3. DuckDB Query
        query = f"""
            SELECT
                m.study_time,
                m.subject_id,
                CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM procedureevents p
                        WHERE p.subject_id = m.subject_id
                          AND p.itemid IN {vent_ids_sql}
                          AND p.starttime > m.study_time
                          AND p.starttime <= m.study_time + INTERVAL '{self.timedelta}'
                    ) THEN 1
                    ELSE 0
                END as ventilation_24h
            FROM
                metadata m
            WHERE
                m.study_time IS NOT NULL
                AND m.subject_id IS NOT NULL
        """

        df = duckdb.sql(query).df()

        # 4. Formatting
        df["study_time"] = pd.to_datetime(df["study_time"])
        df["ventilation_24h"] = df["ventilation_24h"].astype(int)

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

# --- EXECUTION ---
if __name__ == "__main__":
    print("\n--- Initializing Ventilation Prediction Task (Corrected) ---")
    vent_task = CXRVentilationTask(dataset)

    print("--- Generating Ground Truth Labels ---")
    vent_table = vent_task.make_table(mimic_db, pd.Series([dataset.test_timestamp]))

    # --- STATISTICS ---
    total_samples = len(vent_table.df)
    positive_samples = vent_table.df['ventilation_24h'].sum()
    prevalence = (positive_samples / total_samples) * 100

    print(f"\nTotal Study-Patient Pairs: {total_samples}")
    print(f"Positive Cases (Ventilation in <24h): {positive_samples}")
    print(f"Prevalence: {prevalence:.2f}%")
