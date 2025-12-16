from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Dict

import gspread
from google.oauth2.service_account import Credentials

SHEET_ID = "1amdvkApexTuIh8lY5hj1Z9UAn2lC9lz6Fzed9pmEyCw"
WORKSHEET_NAME = "ชีต1"
CREDS_PATH = "service_account.json"

@dataclass
class SheetLogger:
    sheet_id: str = SHEET_ID
    worksheet_name: str = WORKSHEET_NAME
    creds_path: str = CREDS_PATH

    def __post_init__(self):
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(self.creds_path, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(self.sheet_id)          # ✅ ไม่ต้องใช้ Drive API
        self.ws = sh.worksheet(self.worksheet_name)

    def append_event(self, vehicle_id: str, speed_kmh: float, overspeed: bool, message: str):
        row = [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            str(vehicle_id),
            round(float(speed_kmh), 2),
            "YES" if overspeed else "NO",
            message or "",
        ]
        self.ws.append_row(row, value_input_option="USER_ENTERED")


class SpeedRuleEngine:
    def __init__(self, logger: SheetLogger, speed_limit: float = 70.0, cooldown_sec: float = 3.0):
        self.logger = logger
        self.speed_limit = float(speed_limit)
        self.cooldown_sec = float(cooldown_sec)
        self._last_sent: Dict[str, float] = {}

    def push_if_overspeed(self, vehicle_id: str, speed_kmh: Optional[float], message: str = ""):
        if speed_kmh is None:
            return

        vehicle_id = str(vehicle_id)
        speed_kmh = float(speed_kmh)
        if speed_kmh <= self.speed_limit:
            return

        t = time.time()
        last = self._last_sent.get(vehicle_id, 0.0)
        if t - last < self.cooldown_sec:
            return

        self.logger.append_event(
            vehicle_id=vehicle_id,
            speed_kmh=speed_kmh,
            overspeed=True,
            message=message or f"Overspeed > {self.speed_limit:.0f} km/h",
        )
        self._last_sent[vehicle_id] = t
        print(f"[SHEET] id={vehicle_id} speed={speed_kmh:.1f} km/h")
