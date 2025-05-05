import os
import subprocess
import sys

def run_streamlit_app(filename="lead_scoring_streamlit.py"):
    # Get current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, filename)

    # Check if the file exists
    if not os.path.exists(app_path):
        print(f"Error: {filename} not found in {current_dir}")
        sys.exit(1)

    # Run the Streamlit app
    subprocess.run(["streamlit", "run", app_path])

if __name__ == "__main__":
    file = "lead_scoring_streamlit.py"
    run_streamlit_app(file)
