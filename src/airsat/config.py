"""Project-wide constants: paths, seeds, column groups, service groups and segments."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "raw"                      # Kaggle S6E10 competition files
REAL_FILE = DATA_DIR / "original_official" / "data.csv"  # arseniyshutko/binary-aviation-satisfaction-129k
ARTIFACTS_DIR = ROOT / "artifacts"
FOLDS_DIR = ARTIFACTS_DIR / "folds"
OOF_DIR = ARTIFACTS_DIR / "oof"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

SEED = 42
N_FOLDS = 5

TARGET = "satisfaction"
ID_COL = "id"

# Fixed category vocabularies so synthetic and real data always get identical codes.
CATEGORY_LEVELS: dict[str, list[str]] = {
    "Gender": ["Female", "Male"],
    "Customer Type": ["Loyal Customer", "disloyal Customer"],
    "Type of Travel": ["Business travel", "Personal Travel"],
    "Class": ["Business", "Eco", "Eco Plus"],
}
CATEGORICAL = list(CATEGORY_LEVELS)
SERVICE_RATINGS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness",
]
DELAYS = ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
NUMERIC = ["Age", "Flight Distance", *DELAYS]
FEATURES = NUMERIC + SERVICE_RATINGS + CATEGORICAL
EXACT_VALUE_COLUMNS = ["Flight Distance", "Age", *DELAYS]

# Related services grouped for grouped permutation importance (RQ3).
SERVICE_GROUPS: dict[str, list[str]] = {
    "Digital": ["Inflight wifi service", "Ease of Online booking", "Online boarding"],
    "Comfort": ["Seat comfort", "Leg room service", "Cleanliness", "Inflight entertainment", "Food and drink"],
    "Ground": ["Checkin service", "Baggage handling", "Gate location"],
    "Crew": ["On-board service"],
    "Schedule": ["Departure/Arrival time convenient"],
    "Delays": DELAYS,
    "Profile": ["Age", "Gender", "Customer Type", "Type of Travel", "Class", "Flight Distance"],
}

# Six traveller segments: cabin class x trip purpose.
SEGMENT_COLUMNS = ("Class", "Type of Travel")

DISPLAY_NAMES = {
    "Inflight wifi service": "Wi-Fi", "Departure/Arrival time convenient": "Time convenience",
    "Ease of Online booking": "Online booking", "Gate location": "Gate location",
    "Food and drink": "Food & drink", "Online boarding": "Online boarding",
    "Seat comfort": "Seat comfort", "Inflight entertainment": "Entertainment",
    "On-board service": "On-board service", "Leg room service": "Leg room",
    "Baggage handling": "Baggage", "Checkin service": "Check-in", "Cleanliness": "Cleanliness",
    "Type of Travel": "Travel purpose", "Customer Type": "Loyalty", "Class": "Cabin class",
    "Flight Distance": "Flight distance", "Age": "Age", "Gender": "Gender",
    "Departure Delay in Minutes": "Departure delay", "Arrival Delay in Minutes": "Arrival delay",
}
