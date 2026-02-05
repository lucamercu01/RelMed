from relbench.base import Database, Dataset, Table
import io
import torch
import torch.nn as nn
import numpy as np
import torchvision
from PIL import Image
from tqdm import tqdm
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor
import requests
import skimage
import torchxrayvision as xrv


def verify_mimic_access() -> None:
    """Verify that the user has proper access to MIMIC-IV dataset through PhysioNet
    credentialing.

    Verification is done by attempting a small query to the dataset.
    """
    print("Verifying MIMIC-IV access...")
    try:
        from google.cloud import bigquery

        table_id = "physionet-data.mimiciv_3_1_hosp.patients"
        project = os.getenv("PROJECT_ID")
        client = bigquery.Client(project=project)
        client.get_table(table_id)
        client.query("SELECT 1").result()
        print("MIMIC-IV access verified.")
    except Exception as e:
        raise RuntimeError(
            f"\nACCESS FAILED - BigQuery credential check encountered an error: {e}"
        )
def find_time_col(columns):
    for col in [
        "admittime",
        "startdate",
        "chartdate",
        "transfertime",
        "charttime",
        "intime",
        "starttime",
        "study_time",#new time col for metadata
    ]:
        if col in columns:
            return col
    return None
  
def format_ids(series: pd.Series) -> list:
    return list(map(int, series.dropna().unique()))
  
def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Define explicit timestamp columns to protect
    time_cols = [
        "admittime", "dischtime", "deathtime", "edouttime", "edregtime",
        "intime", "outtime", "charttime", "storetime",
        "starttime", "endtime", "study_time", "study_datetime"
    ]
    
    # 2. Convert standard types first
    dtype_mapping = {
        "Int64": int,
        "Int32": int,
        "Float64": float,
        "boolean": bool,
        "object": str,
    }
    
    for col in df.columns:
        # SKIP time columns in the general loop to avoid accidental zero-filling
        if col in time_cols or "time" in col.lower() or "date" in col.lower():
            continue
            
        original_dtype = str(df[col].dtype)
        
        if original_dtype in dtype_mapping:
            # Fill NA for numbers
            if dtype_mapping[original_dtype] in (int, float, complex):
                df[col] = df[col].fillna(0)
            try:
                df[col] = df[col].astype(dtype_mapping[original_dtype])
            except:
                pass # Keep original if cast fails

    # 3. Handle Time Columns Explicitly
    for col in df.columns:
        if col in time_cols or "time" in col.lower() or "date" in col.lower():
            # Force conversion to datetime, handling errors carefully
            # We do NOT fillna(0) here because that creates 1970 dates!
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    return df

def filter_chartevents(table):
    print("Filtering chartevents to only include numeric values")
    chart_df = table.df

    numeric_mask = chart_df["valuenum"].notnull() & (
        chart_df["value"].astype(str) == chart_df["valuenum"].astype(str)
    )
    chartevents_numeric = chart_df[numeric_mask].copy()
    chartevents_numeric = chartevents_numeric.drop(columns=["value"])
    chartevents_numeric["valuenum"] = pd.to_numeric(
        chartevents_numeric["valuenum"], errors="coerce"
    )
    return Table(
        df=chartevents_numeric,
        fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
        pkey_col=table.pkey_col,
        time_col=table.time_col,
    )


def download_and_process_one(args):
    """
    Helper function to download and preprocess for TorchXRayVision.
    FIXED: Now expects only 2 arguments (blob_path, bucket).
    """
    # --- FIX IS HERE: Unpack only 2 arguments ---
    blob_path, bucket = args 
    
    try:
        blob = bucket.blob(blob_path)
        img_bytes = blob.download_as_bytes()
        
        # 1. Load as Numpy Array via PIL
        image = Image.open(io.BytesIO(img_bytes))
        img_array = np.array(image)

        # 2. Handle Dimensions (Convert RGB/RGBA to Grayscale 2D)
        if len(img_array.shape) > 2:
            img_array = img_array.mean(2) # Average channels
        
        # 3. Normalize to [-1024, 1024] range (XRV specific)
        img_array = xrv.datasets.normalize(img_array, 255)
        
        # 4. Convert to Tensor: (1, H, W)
        img_tensor = torch.from_numpy(img_array).float().unsqueeze(0)
        
        # 5. Resize and Crop (224x224)
        img_tensor = torchvision.transforms.Resize(224)(img_tensor)
        img_tensor = torchvision.transforms.CenterCrop(224)(img_tensor)
        
        return img_tensor
        
    except Exception as e:
        # print(f"Error on {blob_path}: {e}") 
        return None

def generate_image_embeddings_gcs(metadata_df, bucket_name, project_id, batch_size=64, num_workers=10):
    """
    Uses TorchXRayVision (DenseNet121-all) to generate medical-grade embeddings.
    Output Dimension: 1024
    """
    print(f"Connessione GCS: {bucket_name} (Progetto: {project_id})")
    print(f"Configurazione: {num_workers} Workers, Batch {batch_size}")
    print("Modello: TorchXRayVision DenseNet-121 (Medical Weights)")

    # 1. Setup GCS con Connection Pool
    storage_client = storage.Client(project=project_id)
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=num_workers,
        pool_maxsize=num_workers,
        max_retries=3
    )
    storage_client._http.mount("https://", adapter)
    storage_client._http.mount("http://", adapter)
    bucket = storage_client.bucket(bucket_name, user_project=project_id)

    # 2. Setup Modello XRV
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    # Initialize the model correctly using .features
    full_model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model = full_model.features 
    model.to(device)
    model.eval()

    all_embeddings = []

    # 3. Prepariamo i percorsi
    paths = []
    subject_ids = metadata_df['subject_id'].astype(str).values
    study_ids = metadata_df['study_id'].astype(str).values
    dicom_ids = metadata_df['dicom_id'].astype(str).values

    for subj, study, dicom in zip(subject_ids, study_ids, dicom_ids):
        blob_path = f"files/p{subj[:2]}/p{subj}/s{study}/{dicom}.jpg"
        paths.append(blob_path)

    print(f"Inizio elaborazione di {len(paths)} immagini...")

    # 4. Esecuzione Parallela
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i in tqdm(range(0, len(paths), batch_size), desc="Processing Batches"):

            batch_paths = paths[i : i + batch_size]
            
            # Pack only 2 arguments to match the helper function
            args = [(p, bucket) for p in batch_paths]

            # Scarica in parallelo
            batch_tensors_list = list(executor.map(download_and_process_one, args))

            valid_tensors = []
            valid_indices = []

            for idx, t in enumerate(batch_tensors_list):
                if t is not None:
                    valid_tensors.append(t)
                    valid_indices.append(idx)

            if not valid_tensors:
                batch_embeddings = np.zeros((len(batch_paths), 1024))
                all_embeddings.extend(batch_embeddings)
                continue

            # Stack e Inferenza
            batch_input = torch.stack(valid_tensors).to(device)

            with torch.no_grad():
                features = model(batch_input)
                
                # Pool (Batch, 1024, 7, 7) -> (Batch, 1024)
                if len(features.shape) == 4:
                    features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                    features = features.view(features.size(0), -1)
                
                features = features.cpu().numpy()

            if len(features.shape) == 1:
                features = features.reshape(1, -1)

            # Riordino
            final_batch_embeddings = []
            valid_ptr = 0
            for j in range(len(batch_paths)):
                if j in valid_indices:
                    final_batch_embeddings.append(features[valid_ptr])
                    valid_ptr += 1
                else:
                    final_batch_embeddings.append(np.zeros(1024))

            all_embeddings.extend(final_batch_embeddings)

    return np.array(all_embeddings)

class MimicDataset(Dataset):
    """A dataset class for loading and processing MIMIC-IV and MIMIMC-CXR-jpg data into a format suitable
    for use with RelBench. This class extracts a subset of tables specified in
    `tables_limit`, filters the data based on ICU stay and patient criteria, and allows
    optional column dropouts to reduce dimensionality or remove irrelevant fields.

    The dataset is anchored around ICU stays (`icustays`) and patients (`patients`) and constructs all other
    tables with respect to these entities.

    If certain parameters are not provided, default values will be used.

    Parameters:
        project_id (str): Google Cloud project ID for BigQuery.
        patients_limit (int): Maximum number of patients to include in the dataset (0 means no limit; defaults to 20 000 if not specified).
        saving_data (bool): Whether to persist processed tables to disk as .H5 files (default: True).
        out_path (str): Output directory to save HDF5 files (default: "data").
        cache_dir (str): Directory used for caching dataset.
        tables_limit (list): List of table names to include in the dataset (default: common MIMIC-IV tables).
        drop_columns_per_table (dict): Dictionary specifying columns to drop per table (default: predefined subset per table).
        min_age (int): Minimum patient age in years to include (default: 15).
        min_dur (int): Minimum ICU stay duration in hours (default: 36).
        max_dur (int): Maximum ICU stay duration in hours (default: 240).
        location (str): Location for BigQuery client (default: 'US').
        dataset_name (str): Name of the BigQuery dataset (default: 'physionet-data').

    Example:
        drop_columns_per_table = {
            "admissions": [...],
            "chartevents": [...],
            ...
        }
        tables_limit = ["patients", "admissions", "icustays", "chartevents", "procedureevents", "d_items"]

        dataset = MimicDataset(
            patients_limit=20 000,
            out_path='/data',
            cache_dir='/cache',
            tables_limit=tables_limit,
            db_params=db_params,
            drop_columns_per_table=drop_columns_per_table
        )
        db = dataset.make_db()
    """

    def __init__(
        self,
        project_id: str = None,
        patients_limit: int = -1,
        saving_data: bool = True,
        out_path: str = "data",
        cache_dir: str = None,
        tables_limit: list = None,
        drop_columns_per_table=None,
        min_age: int = 15,
        min_dur: int = 36,
        max_dur: int = 240,
        location: str = "US",
        dataset_name: str = "physionet-data",
        bucket_name: str = "mimic-cxr-jpg-2.1.0.physionet.org"
    ):
        # Lazy import to avoid requiring google-cloud-bigquery for other datasets
        from google.cloud import bigquery

        super().__init__(cache_dir=cache_dir)

        # Load environment variables from .env file (lazy import for optional dependency)
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            # dotenv is optional - if not available, env vars can still be set manually
            pass

        # Use environment variable if project_id not provided
        if project_id is None:
            try:
                project_id = os.getenv("PROJECT_ID")
            except:
                raise ValueError(
                    "project_id is required: set MIMIC_BQ_PROJECT_ID environment variable (i.e. export MIMIC_BQ_PROJECT_ID='your-project_id')"
                )

        if drop_columns_per_table is None:
            print(f"drop_columns_per_table not provided, dropping default.")
            drop_columns_per_table = {
                "admissions": [
                    "admittime",
                    "dischtime",
                    "deathtime",
                    "edouttime",
                    "edregtime",
                    "hospital_expire_flag",
                    "discharge_location",
                    "race",
                ],
                "chartevents": ["storetime"],
                "procedureevents": [
                    "storetime",
                    "endtime",
                    "location",
                    "location_category",
                    "location",
                    "locationcategory",
                    "linkorderid",
                    "originalamount",
                    "originalrate",
                ],
                "d_items": ["abbreviation"],
                # NEW: CXR drops:we keep subject_id, study_id, and the labels.
                # We drop dicom_id if we don't need the images for RelBench (purely tabular task)
                "metadata": [
                    #"dicom_id",
                    "ViewPosition",
                    "Rows",
                    "Columns"
                    ],
            }

        if tables_limit is None:
            print(f"tables_limit not provided setting to default.")
            tables_limit = [
                "patients",
                "admissions",
                "icustays",
                "chartevents",
                "procedureevents",
                "d_items",
                "metadata", # Contains StudyDate/Time
                "chexpert",  # Contains Labels
            ]

        if patients_limit == -1:
            patients_limit = 20000
            print(f"patients_limit not provided setting to default {patients_limit}.")

        self.query_args = [
            bigquery.ScalarQueryParameter("limit", "INT64", patients_limit),
            bigquery.ScalarQueryParameter("min_age", "INT64", min_age),
            bigquery.ScalarQueryParameter("min_dur", "INT64", min_dur),
            bigquery.ScalarQueryParameter("max_dur", "INT64", max_dur),
            bigquery.ScalarQueryParameter("min_day", "FLOAT64", float(min_dur) / 24),
        ]

        self.saving_data = saving_data
        self.patients_limit = patients_limit
        self.drop_columns_per_table = drop_columns_per_table
        self.bucket_name = bucket_name
        self.tables_limit = tables_limit

        # If a cached RelBench copy exists (e.g., downloaded db.zip), reuse it and
        # avoid hitting BigQuery.
        cached_db_dir = Path(cache_dir) / "db" if cache_dir is not None else None
        if cached_db_dir and cached_db_dir.exists() and any(cached_db_dir.iterdir()):
            print(
                f"Found cached MIMIC database at {cached_db_dir}, skipping BigQuery build."
            )
            cached_db = Database.load(cached_db_dir)
            self._set_timestamps_from_icustays(cached_db.table_dict["icustays"].df)
            return

        # Create the output directory if it doesn't exist
        current_dir = os.getcwd()
        self.out_path = Path(current_dir) / out_path / f"limit_{patients_limit}"
        if saving_data:
            self.out_path.mkdir(parents=True, exist_ok=True)
            print("Data will be saved to", self.out_path)

        self.client = bigquery.Client(project=project_id, location=location)
        self.dataset_name = dataset_name

        # Set the test and validation timestamps
        # this will be updated based on patients' ICU admission times
        self.test_timestamp = None
        self.val_timestamp = None

       # print("Making database in __init__....")
        #self.make_db()

    def _set_timestamps_from_icustays(self, icustays_df: pd.DataFrame) -> None:
        timestamps = (
            pd.to_datetime(icustays_df["intime"], errors="coerce")
            .dropna()
            .sort_values()
        )
        if len(timestamps) == 0:
            raise ValueError("Unable to set timestamps: icustays.intime is empty.")

        val_idx = int(len(timestamps) * 0.7)
        test_idx = int(len(timestamps) * 0.85)
        self.val_timestamp = timestamps.iloc[val_idx]
        self.test_timestamp = timestamps.iloc[test_idx]

    def make_db(self) -> Database:
        from google.cloud import bigquery

        start_time = time.time()
        tables_df = {}
        # while MIMIC IV dataset on BigQuery does not have primary and foreign keys set up properly (at all)
        # we need to declare the relations statically here..
        table_names_schemas = {
            "mimiciv_3_1_hosp": [["patients"], ["admissions"]],
            "mimiciv_3_1_icu": [
                ["icustays"],
                ["procedureevents"],
                ["d_items"],
                ["chartevents"],
            ],
            "mimic_cxr_jpg": [
                ["metadata"],
                ["chexpert"],
            ]
        }
        table_key_map = {
            "patients": {"pkey_col": "subject_id", "fkey_col_to_pkey_table": {}},
            "icustays": {
                "pkey_col": "stay_id",
                "fkey_col_to_pkey_table": {
                    "hadm_id": "admissions",
                    "subject_id": "patients",
                },
            },
            "admissions": {
                "pkey_col": "hadm_id",
                "fkey_col_to_pkey_table": {"subject_id": "patients"},
            },
            "procedureevents": {
                "pkey_col": "orderid",
                "fkey_col_to_pkey_table": {
                    "hadm_id": "admissions",
                    "itemid": "d_items",
                    "stay_id": "icustays",
                    "subject_id": "patients",
                },
            },
            "d_items": {"pkey_col": "itemid", "fkey_col_to_pkey_table": {}},
            "chartevents": {
                "pkey_col": None,
                "fkey_col_to_pkey_table": {
                    "hadm_id": "admissions",
                    "itemid": "d_items",
                    "stay_id": "icustays",
                    "subject_id": "patients",
                },
            },
            # --- NEW: CXR Key Mappings ---
            # Metadata is the "root" of the study info, containing the timestamp
            "metadata": {
                "pkey_col": "study_id",
                "fkey_col_to_pkey_table": {
                    "subject_id": "patients"
                },
            },
            # CheXpert contains the target labels for the study
            "chexpert": {
                "pkey_col": None,
                "fkey_col_to_pkey_table": {
                    "subject_id": "patients",
                    "study_id": "metadata"
                },
            },
        }

        tables_df["patients"] = Table(
            df=self.get_patients(),
            fkey_col_to_pkey_table=table_key_map["patients"]["fkey_col_to_pkey_table"],
            pkey_col=table_key_map["patients"]["pkey_col"],
            time_col=None,
        )

        query_params = [
            bigquery.ArrayQueryParameter(
                "subject_ids",
                "INT64",
                format_ids(tables_df["patients"].df["subject_id"]),
            ),
            bigquery.ArrayQueryParameter(
                "stay_ids", "INT64", format_ids(tables_df["patients"].df["stay_id"])
            ),
            bigquery.ArrayQueryParameter(
                "hadm_ids", "INT64", format_ids(tables_df["patients"].df["hadm_id"])
            ),
        ]

        tables_df["icustays"] = Table(
            df=self.get_icustays(query_params),
            fkey_col_to_pkey_table=table_key_map["icustays"]["fkey_col_to_pkey_table"],
            pkey_col=table_key_map["icustays"]["pkey_col"],
            time_col="intime",
        )
        query_params = [
            bigquery.ArrayQueryParameter(
                "subject_ids",
                "INT64",
                format_ids(tables_df["icustays"].df["subject_id"]),
            ),
            bigquery.ArrayQueryParameter(
                "stay_ids", "INT64", format_ids(tables_df["icustays"].df["stay_id"])
            ),
            bigquery.ArrayQueryParameter(
                "hadm_ids", "INT64", format_ids(tables_df["icustays"].df["hadm_id"])
            ),
        ]

        for schema, tables in table_names_schemas.items():
            for (table_name,) in tables:
                if table_name in {"patients", "icustays"}:
                    continue
                h5_path = self.out_path / f"{table_name}_{self.patients_limit}.H5"
                if not h5_path.exists():
                    print(f"Creating {table_name}")

                    # Create the query string
                    query_string = self.build_query(table_name, schema)

                    print("Querying table:", table_name, end=" ")
                    # Execute the query
                    df = self.query(
                        query_string=query_string, query_params=query_params
                    )
                    # [FIX START] Referential Integrity Filter for CheXpert
                    if table_name == "chexpert":
                        # Ensure metadata is already loaded (it usually is due to dictionary order)
                        if "metadata" in tables_df:
                            valid_studies = set(tables_df["metadata"].df["study_id"])
                            
                            # Calculate how many we are dropping
                            initial_len = len(df)
                            df = df[df["study_id"].isin(valid_studies)]
                            dropped_len = initial_len - len(df)
                            
                            print(f"Filtered chexpert: Dropped {dropped_len} rows with no corresponding metadata.")
                    # [FIX END]
                    print(f"{len(df)} rows")

                    # Drop columns if specified
                    if (
                        self.drop_columns_per_table
                        and table_name in self.drop_columns_per_table
                    ):
                        df = self.drop_columns(table_name, df)
                    if table_name == "metadata":
                        if self.saving_data:
                            # Definiamo il percorso del file di cache su Drive
                            # self.out_path è già su /content/drive/MyDrive/...
                            emb_cache_path = self.out_path / "cxr_embeddings_cache.pt"

                            # 1. Controlla se il file esiste già
                            if emb_cache_path.exists():
                                print(f"✅ Trovata cache embeddings in: {emb_cache_path}")
                                print("Caricamento embeddings senza ricalcolare...")
                                
                                # Carichiamo il dizionario {study_id: embedding}
                                emb_dict = torch.load(emb_cache_path)
                                
                                # Funzione helper per recuperare l'embedding
                                def get_emb_safe(sid):
                                    # Se l'ID esiste nella cache, usalo. 
                                    # Altrimenti (nuovi pazienti?) metti zero.
                                    if sid in emb_dict:
                                        return emb_dict[sid]
                                    else:
                                        return np.zeros(1024, dtype=np.float32)

                                # Applichiamo il mapping usando study_id come chiave
                                df['image_embedding'] = df['study_id'].apply(get_emb_safe)
                                print("Embeddings caricati e mappati correttamente.")

                            else:
                                # 2. Se non esiste, calcolali da zero
                                print("Cache non trovata. Generazione embedding da GCS (richiede tempo)...")
                                
                                project_id = self.client.project
                                image_features = generate_image_embeddings_gcs(
                                    df,
                                    self.bucket_name,
                                    project_id=project_id
                                )

                                # Convertiamo in lista per il DataFrame
                                list_features = list(image_features)
                                df['image_embedding'] = list_features
                                
                                # 3. SALVATAGGIO: Creiamo il dizionario e salviamo su Drive
                                print(f"Salvataggio cache in {emb_cache_path}...")
                                # Creiamo una mappa ID -> Vettore
                                emb_dict = dict(zip(df['study_id'], list_features))
                                torch.save(emb_dict, emb_cache_path)
                                print("✅ Cache salvata con successo. Al prossimo avvio verrà caricata automaticamente.")


                    # Save the DataFrame to HDF5 if saving_data is True
                    if self.saving_data:
                        # Defensive: ensure the file is not lingering or locked
                        if h5_path.exists():
                            try:
                                h5_path.unlink()
                                print(f"Cleaned up existing (possibly locked) file: {h5_path}")
                            except OSError as e:
                                print(f"ERROR: Could not remove {h5_path} before writing: {e}")
                                raise e

                        df.to_hdf(h5_path, key="table", index=False)
                        print(f"Table {table_name} saved to {h5_path}")
                else:
                    print(
                        f"️File {table_name}_{self.patients_limit}.H5 already exists. Skipping..."
                    )
                    df = pd.read_hdf(h5_path, key="table")

                # Create the Table object for the current table
                tables_df[table_name] = Table(
                    df=df,
                    fkey_col_to_pkey_table=table_key_map[table_name][
                        "fkey_col_to_pkey_table"
                    ],
                    pkey_col=table_key_map[table_name]["pkey_col"],
                    time_col=find_time_col(df.columns),
                )

        # Filter chartevents to only include numeric values
        tables_df["chartevents"] = filter_chartevents(tables_df["chartevents"])

        ''' VECCHIO CALCOLO DI TEST E VAL TIMESTAMPS BASATO SU ICUSTAYS
        # Change test and val timestamps based on patients limit
        print("Setting test and val timestamps based on patients limit")
        timestamps = (
            pd.to_datetime(tables_df["icustays"].df["intime"], errors="coerce")
            .dropna()
            .sort_values()
        )
        n = len(timestamps)

        # Calculate the indices for 70% and 85%
        # of the total number of timestamps
        # 70% for train and 15% for validation and 15% for test
        val_idx = int(n * 0.7)
        test_idx = int(n * 0.85)
        self.val_timestamp = timestamps.iloc[val_idx]
        self.test_timestamp = timestamps.iloc[test_idx]
        train_count = len(timestamps[timestamps < self.val_timestamp])
        val_count = len(
            timestamps[
                (timestamps >= self.val_timestamp) & (timestamps < self.test_timestamp)
            ]
        )
        test_count = len(timestamps[timestamps >= self.test_timestamp])
        print("test_timestamp:", self.test_timestamp, end=" ")
        print("val_timestamp:", self.val_timestamp)
        print(
            "Record split - Train:", train_count, "Val:", val_count, "Test:", test_count
        )

        end_time = time.time()
        print(f"Done loading tables in {end_time - start_time:.2f} seconds")
        return Database(tables_df)
        '''
        # C. CALCOLO TIMESTAMPS BASATO SU CXR (Punto 2)
        print("Setting test and val timestamps based on CXR STUDY TIME")

        # Prendiamo i tempi dalla tabella metadata, non da icustays

        timestamps = (
            pd.to_datetime(tables_df["metadata"].df["study_time"], errors="coerce")
            .dropna()
            .sort_values()
        )

        if len(timestamps) == 0:
             raise ValueError("Nessun timestamp trovato in metadata. Controlla la creazione della colonna study_time.")

        # Calculate the indices for 70% and 85%
        # of the total number of timestamps
        # 70% for train and 15% for validation and 15% for test

        n = len(timestamps)
        val_idx = int(n * 0.7)
        test_idx = int(n * 0.85)

        self.val_timestamp = timestamps.iloc[val_idx]
        self.test_timestamp = timestamps.iloc[test_idx]

        # Statistiche di controllo
        train_count = len(timestamps[timestamps < self.val_timestamp])
        val_count = len(timestamps[(timestamps >= self.val_timestamp) & (timestamps < self.test_timestamp)])
        test_count = len(timestamps[timestamps >= self.test_timestamp])

        print("CXR Split - Test Time:", self.test_timestamp, "Val Time:", self.val_timestamp)
        print(f"Studies split - Train: {train_count}, Val: {val_count}, Test: {test_count}")
        # Forzatura conversione
        tables_df["metadata"].df["study_time"] = pd.to_datetime(tables_df["metadata"].df["study_time"])


        end_time = time.time()
        print(f"Done loading tables in {end_time - start_time:.2f} seconds")
        return Database(tables_df)

    def get_patients(self):
        patients_filename = "patients_" + str(self.patients_limit) + ".H5"

        # Query the patients
        ''' VECCHIA QUERY
        patients_query = f"""
        SELECT
            i.subject_id,                  -- Unique patient identifier
            i.hadm_id,                     -- Unique hospital admission ID
            i.stay_id,                     -- Unique ICU stay ID
            i.gender,                      -- Patient gender
            ROUND(i.admission_age) AS age, -- Age of the patient at hospital admission
            i.race                         -- Patient race/ethnicity
        FROM `{self.dataset_name}.mimiciv_3_1_derived.icustay_detail` i
        WHERE i.hadm_id IS NOT NULL
           AND i.stay_id IS NOT NULL
           AND i.hospstay_seq = 1
           AND i.icustay_seq = 1
           AND i.los_icu IS NOT NULL
           AND i.admission_age >= @min_age
           AND i.los_icu >= @min_day
           AND (UNIX_SECONDS(TIMESTAMP(i.icu_outtime)) - UNIX_SECONDS(TIMESTAMP(i.icu_intime))) > (@min_dur*3600)
           AND (UNIX_SECONDS(TIMESTAMP(i.icu_outtime)) - UNIX_SECONDS(TIMESTAMP(i.icu_intime))) < (@max_dur*3600)
           AND EXISTS (
             SELECT 1
             FROM `{self.dataset_name}.mimiciv_3_1_icu.icustays` icu
             WHERE icu.stay_id = i.stay_id
             )
        QUALIFY
            ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.icustay_seq DESC) = 1
        ORDER BY subject_id
        """
        '''
        # NUOVA QUERY in cui aggiungo che il paziente deve avere almeno una radiografia eseguita
        patients_query = f"""
        SELECT
            i.subject_id,                  -- Unique patient identifier
            i.hadm_id,                     -- Unique hospital admission ID
            i.stay_id,                     -- Unique ICU stay ID
            i.gender,                      -- Patient gender
            ROUND(i.admission_age) AS age, -- Age of the patient at hospital admission
            i.race                         -- Patient race/ethnicity
        FROM `{self.dataset_name}.mimiciv_3_1_derived.icustay_detail` i
        WHERE i.hadm_id IS NOT NULL
           AND i.stay_id IS NOT NULL
           AND i.hospstay_seq = 1
           AND i.icustay_seq = 1
           AND i.los_icu IS NOT NULL
           AND i.admission_age >= @min_age
           AND i.los_icu >= @min_day
           AND (UNIX_SECONDS(TIMESTAMP(i.icu_outtime)) - UNIX_SECONDS(TIMESTAMP(i.icu_intime))) > (@min_dur*3600)
           AND (UNIX_SECONDS(TIMESTAMP(i.icu_outtime)) - UNIX_SECONDS(TIMESTAMP(i.icu_intime))) < (@max_dur*3600)
           AND EXISTS (
               SELECT 1
               FROM `{self.dataset_name}.mimic_cxr_jpg.metadata` cxr
               WHERE cxr.subject_id = i.subject_id
             )
        QUALIFY
            ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.icustay_seq DESC) = 1
        ORDER BY subject_id
        """
        if self.patients_limit > 0:
            patients_query += "LIMIT @limit"

        H5_fpath = os.path.join(self.out_path, patients_filename)
        if not os.path.exists(H5_fpath):
            patients = self.query(
                query_string=patients_query, query_params=self.query_args
            )
            if self.saving_data:
                patients.to_hdf(H5_fpath, key="patients", index=False)
                print(f"Patients data saved to {H5_fpath}")
        else:
            print(f"File {patients_filename} already exists. Skipping...")
            patients = pd.read_hdf(H5_fpath, key="patients")
        return patients

    def get_icustays(self, params: list):
        icustays_filename = "icustays_" + str(self.patients_limit) + ".H5"
        H5_fpath = os.path.join(self.out_path, icustays_filename)
        # Query the ICU_Stay
        icustays_query = f"""
         SELECT icu.subject_id,
                icu.hadm_id,
                icu.stay_id,
                icu.intime,
                icu.first_careunit,
                icu_detail.los_icu
         FROM `{self.dataset_name}.mimiciv_3_1_icu.icustays` AS icu
              LEFT JOIN `{self.dataset_name}.mimiciv_3_1_derived.icustay_detail` AS icu_detail
              ON icu.stay_id = icu_detail.stay_id
         WHERE icu.subject_id IN UNNEST(@subject_ids)
            AND icu.stay_id IN UNNEST(@stay_ids)
            AND icu.hadm_id IN UNNEST(@hadm_ids)
            AND icu_detail.los_icu IS NOT NULL
         ORDER BY icu.subject_id
         """
        if not os.path.exists(H5_fpath):
            icustays = self.query(query_string=icustays_query, query_params=params)
            icustays["los_icu"] = pd.to_numeric(icustays["los_icu"], errors="coerce")
            if self.saving_data:
                icustays.to_hdf(H5_fpath, key="icustays", index=False)
                print(f"ICUStays data saved to {H5_fpath}")
        else:
            print(f"File {icustays_filename} already exists. Skipping...")
            icustays = pd.read_hdf(H5_fpath, key="icustays")

        return icustays

    def build_query(self, table_name, schema):
        columns = self.get_columns(table_name, schema)
        conditions = []

        # Helper to find column names regardless of case (e.g., Study_ID vs study_id)
        subj_col = next((c for c in columns if c.lower() == "subject_id"), None)
        hadm_col = next((c for c in columns if c.lower() == "hadm_id"), None)
        stay_col = next((c for c in columns if c.lower() == "stay_id"), None)
        # Crucial: Find the actual study_id column name in BigQuery
        actual_study_col = next((c for c in columns if c.lower() == "study_id"), None)

        # Build conditions dynamically based on found columns
        if subj_col:
            conditions.append(f"{subj_col} IN UNNEST(@subject_ids)")
        if stay_col:
            conditions.append(f"{stay_col} IN UNNEST(@stay_ids)")
        if hadm_col:
            conditions.append(f"{hadm_col} IN UNNEST(@hadm_ids)")

        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)
        else:
            where_clause = ""

        # --- LOGIC FIX HERE ---
        if table_name == "metadata":
            # Your existing logic for metadata time parsing
            query_string = f"""
            SELECT
                * EXCEPT(StudyDate, StudyTime),
                PARSE_DATETIME(
                    '%Y%m%d%H%M%S',
                    CONCAT(
                        CAST(StudyDate AS STRING),
                        LPAD(CAST(CAST(StudyTime AS INT64) AS STRING), 6, '0')
                    )
                ) as study_time
            FROM `{self.dataset_name}.{schema}.{table_name}`
            {where_clause}
            QUALIFY ROW_NUMBER() OVER (PARTITION BY study_id ORDER BY StudyDate, StudyTime) = 1
            """

        elif table_name == "chexpert":
          # LOGIC FIX: Use EXCEPT to prevent duplicate columns (study_id and study_id_1)
          # We remove the original column (whatever case it is) and re-select it as lowercase 'study_id'
          query_string = f"""
          SELECT * EXCEPT({actual_study_col}), {actual_study_col} AS study_id
          FROM `{self.dataset_name}.{schema}.{table_name}`
          {where_clause}
          """

        else:
            query_string = f"SELECT * FROM `{self.dataset_name}.{schema}.{table_name}`{where_clause}"

        return query_string


    def drop_columns(self, table_name, df):
        drop_cols = list(
            set(self.drop_columns_per_table[table_name]) & set(df.columns)
        )  # intersection
        if drop_cols:
            df = df.drop(columns=drop_cols)
            print(f"Dropped columns from {table_name}: {drop_cols}")
        return df

    def get_columns(self, table_name, schema):
        table_id = f"{self.dataset_name}.{schema}.{table_name}"
        table = self.client.get_table(table_id)

        # Extract column names from the schema
        column_names = [field.name for field in table.schema]
        return column_names

    def query(self, query_string, query_params: list = []):
        from google.cloud import bigquery

        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        
        # 1. Fetch raw DataFrame from BigQuery
        df = self.client.query_and_wait(
            query_string, job_config=job_config
        ).to_dataframe()

        

        # 2. Apply conversion
        
        df = convert_dtypes(df)

        return df
