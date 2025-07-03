import logging
import os
from datetime import datetime

# Buat folder logs jika belum ada
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Buat nama file log berdasarkan timestamp
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Konfigurasi logging
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
