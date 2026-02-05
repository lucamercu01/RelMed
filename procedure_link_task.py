from relbench.base import TaskType, RecommendationTask, Database, Table
import pandas as pd
import duckdb
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
    mae,
    r2,
    rmse,
    roc_auc,
)
class CXRProcedureLinkTask(RecommendationTask):
    r"""Link Prediction: Predict which specific procedures (from d_items)
    will be performed on the patient within 24 hours of a Chest X-ray."""

    task_type = TaskType.LINK_PREDICTION

    # Source Node (The Patient)
    src_entity_col = "subject_id"
    src_entity_table = "patients"

    # Destination Node (The Procedure Item)
    dst_entity_col = "itemid"
    dst_entity_table = "d_items"

    time_col = "study_time"

    # We look for links formed in this window
    timedelta = pd.Timedelta(hours=24)

    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 12

    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.test_timestamp = dataset.test_timestamp
        self.val_timestamp = dataset.val_timestamp

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
      # 1. Load DataFrames
      metadata = db.table_dict["metadata"].df
      procedureevents = db.table_dict["procedureevents"].df
      d_items = db.table_dict["d_items"].df

      # 2. Create a mapping from raw itemid to internal index (0 to 4094)
      # This ensures the 'itemid' in your task matches the node indices in the graph
      itemid_to_idx = {itemid: i for i, itemid in enumerate(d_items["itemid"])}

      # 3. Join metadata and procedureevents in DuckDB
      query = f"""
          SELECT
              m.study_time,
              m.subject_id,
              p.itemid
          FROM
              metadata m
          JOIN
              procedureevents p ON m.subject_id = p.subject_id
          WHERE
              p.starttime > m.study_time
              AND p.starttime <= m.study_time + INTERVAL '{self.timedelta}'
      """
      raw_df = duckdb.sql(query).df()

      # 4. Map the raw itemid to the internal index
      # If an itemid isn't in our d_items table, we drop it
      raw_df["itemid"] = raw_df["itemid"].map(itemid_to_idx)
      raw_df = raw_df.dropna(subset=["itemid"])
      raw_df["itemid"] = raw_df["itemid"].astype(int)

      # 5. Group by patient and time to create the list of target indices
      df = raw_df.groupby(["study_time", "subject_id"])["itemid"].apply(list).reset_index()

      # 6. Final formatting
      df["study_time"] = pd.to_datetime(df["study_time"])

      return Table(
          df=df,
          fkey_col_to_pkey_table={
              self.src_entity_col: self.src_entity_table,
              self.dst_entity_col: self.dst_entity_table
          },
          pkey_col=None,
          time_col=self.time_col
      )
