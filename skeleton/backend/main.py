from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
import os
import shutil
from dotenv import load_dotenv
from agent.payroll_agent import PayrollAgent
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from datetime import datetime
import threading

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Configure CORS to allow requests from the frontend
origins = [
    "null",  # Allows requests from local files (file://)
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory configurations
INPUT_DIR = "input_files"
PROCESSED_DIR = "processed_files"
ARCHIVE_DIR = "archive"

# Create necessary directories
for dir_path in [INPUT_DIR, PROCESSED_DIR, ARCHIVE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Global variables to track processing status
processing_status = {
    "is_processing": False,
    "last_processed": None,
    "last_report": None,
    "error": None,
    "required_files": {
        "master_data": {"status": "missing", "last_modified": None},
        "pay_register_may": {"status": "missing", "last_modified": None},
        "pay_register_june": {"status": "missing", "last_modified": None},
        "gl_data": {"status": "missing", "last_modified": None},
        "sit_data": {"status": "missing", "last_modified": None}
    }
}

def identify_file_type(filename: str) -> Optional[str]:
    """Identify the type of file based on its name pattern."""
    filename = filename.lower()
    if "master" in filename:
        return "master_data"
    elif "may" in filename and "register" in filename:
        return "pay_register_may"
    elif "june" in filename and "register" in filename:
        return "pay_register_june"
    elif "gl" in filename or "ledger" in filename:
        return "gl_data"
    elif "sit" in filename or "template" in filename:
        return "sit_data"
    return None

def process_files():
    """Process the files when all required files are present."""
    global processing_status
    
    try:
        # Check if all required files are present
        all_files_present = all(
            info["status"] == "present" 
            for info in processing_status["required_files"].values()
        )
        
        if all_files_present and not processing_status["is_processing"]:
            processing_status["is_processing"] = True
            processing_status["error"] = None
            
            # Collect file paths
            file_paths = {}
            for file_type, info in processing_status["required_files"].items():
                matching_files = [f for f in os.listdir(INPUT_DIR) 
                                if identify_file_type(f) == file_type]
                if matching_files:
                    file_paths[file_type] = os.path.join(INPUT_DIR, matching_files[0])
            
            # Get API key from environment
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise Exception("GEMINI_API_KEY not found in environment variables.")
            
            # Initialize and run the agent
            agent = PayrollAgent(gemini_api_key=api_key, file_paths=file_paths)
            report = agent.run_all_checks()
            
            # Update status
            processing_status["last_processed"] = datetime.now().isoformat()
            processing_status["last_report"] = report
            
            # Archive processed files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_subdir = os.path.join(ARCHIVE_DIR, timestamp)
            os.makedirs(archive_subdir, exist_ok=True)
            
            for file_path in file_paths.values():
                filename = os.path.basename(file_path)
                archive_path = os.path.join(archive_subdir, filename)
                shutil.move(file_path, archive_path)
            
            # Reset file status
            for file_type in processing_status["required_files"]:
                processing_status["required_files"][file_type] = {
                    "status": "missing",
                    "last_modified": None
                }
            
    except Exception as e:
        processing_status["error"] = str(e)
        print(f"Error processing files: {e}")
    finally:
        processing_status["is_processing"] = False

class PayrollFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
            
        filename = os.path.basename(event.src_path)
        file_type = identify_file_type(filename)
        
        if file_type:
            processing_status["required_files"][file_type] = {
                "status": "present",
                "last_modified": datetime.now().isoformat()
            }
            
            # Start processing in a separate thread
            threading.Thread(target=process_files).start()
            
    def on_deleted(self, event):
        if event.is_directory:
            return
            
        filename = os.path.basename(event.src_path)
        file_type = identify_file_type(filename)
        
        if file_type:
            processing_status["required_files"][file_type] = {
                "status": "missing",
                "last_modified": None
            }

# Start the file watcher
observer = Observer()
observer.schedule(PayrollFileHandler(), INPUT_DIR, recursive=False)
observer.start()

@app.get("/api/status")
async def get_status():
    """Get the current processing status and required files status."""
    return processing_status

@app.get("/")
def read_root():
    return {"message": "Payroll Agent API is running. Place files in the input_files directory for automatic processing."}
