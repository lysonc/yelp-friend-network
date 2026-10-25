from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "preliminary_results"
EDA = RESULTS / "0_eda"
CITY_SELECTION = RESULTS / "1_city_selection"
COMMUNITIES = RESULTS / "2_communities"
