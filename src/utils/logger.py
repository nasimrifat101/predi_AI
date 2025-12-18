import pandas as pd
from pathlib import Path

LOG_FILE = Path("../../data/logs/game_log.csv")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def log_round(timestamp, round_num, timer, ribbon, raw_file, ribbon_file, round_file, timer_file):
    # Prepare a dataframe row
    df = pd.DataFrame([{
        "timestamp": timestamp,
        "round": round_num,
        "timer": timer,
        "ribbon": ",".join(ribbon),
        "raw_file": str(raw_file),
        "ribbon_file": str(ribbon_file),
        "round_file": str(round_file),
        "timer_file": str(timer_file)
    }])
    
    # Append to CSV
    if LOG_FILE.exists():
        df.to_csv(LOG_FILE, mode='a', index=False, header=False)
    else:
        df.to_csv(LOG_FILE, index=False, header=True)
    
    print(f"[INFO] Round logged: {timestamp}")
