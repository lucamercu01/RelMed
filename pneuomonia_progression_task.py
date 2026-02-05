from relbench.base import TaskType, EntityTask, Database, Table
from relbench.metrics import roc_auc, auprc, accuracy, f1
import duckdb
import pandas as pd

class CXRPneumoniaProgressionTask(EntityTask):
    r"""
    Predict if a patient will develop radiological evidence of Pneumonia 
    within 4 days, based on their current status and X-ray.
    """
    
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "subject_id"
    entity_table = "patients"
    time_col = "study_time"
    target_col = "pneumonia_next_7d"
    timedelta = pd.Timedelta(days=1)
    metrics = [roc_auc, auprc, accuracy, f1]

    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.test_timestamp = dataset.test_timestamp
        self.val_timestamp = dataset.val_timestamp

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        # 1. Load Data
        chexpert = db.table_dict["chexpert"].df
        metadata = db.table_dict["metadata"].df
        
        # 2. Prepare Labels (Clean up the CheXpert labels)
        # 1.0 = Positive, 0.0 = Negative, -1.0 = Uncertain
        # Strategy: Treat 'Uncertain' (-1) as Positive (risk) or Negative. 
        # Standard practice is often U-Ones (treat as 1) or U-Zeros (treat as 0).
        # Here we map -1 -> 1 to capture all potential risks.
        label_df = chexpert[["study_id", "Pneumonia"]].copy()
        label_df["Pneumonia"] = label_df["Pneumonia"].replace(-1.0, 1.0).fillna(0.0)
        
        # 3. Join Metadata to align Time with Labels
        # We need (subject_id, study_time, has_pneumonia)
        joined = metadata.merge(label_df, on="study_id", how="inner")
        
        # Register for DuckDB
        duckdb.register("joined_data", joined)

        # 4. Define the Target (Future Pneumonia)
        # For every scan (m), look ahead 7 days. If ANY future scan shows Pneumonia, target = 1.
        query = f"""
            SELECT
                m.study_time,
                m.subject_id,
                CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM joined_data f
                        WHERE f.subject_id = m.subject_id
                          AND f.study_time > m.study_time
                          AND f.study_time <= m.study_time + INTERVAL '7 days'
                          AND f.Pneumonia = 1.0
                    ) THEN 1
                    ELSE 0
                END as pneumonia_next_7d
            FROM
                joined_data m
            WHERE
                m.study_time IS NOT NULL
        """
        
        df = duckdb.sql(query).df()
        
        # Cleanup
        df["study_time"] = pd.to_datetime(df["study_time"])
        df["pneumonia_next_7d"] = df["pneumonia_next_7d"].astype(int)
        
        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )
