import os
import logging
from ui.app import start_ui

# Ensure output directory
OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Configure logging to a file in the output directory
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.join(OUTPUT_DIR, 'app.log'),
    filemode='a'
)

if __name__ in {'__main__', '__mp_main__'}:
    start_ui()
