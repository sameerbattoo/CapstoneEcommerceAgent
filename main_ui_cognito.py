"""
Main entry point for the E-Commerce Assistant Cloud UI (Cognito as the Inbounf Auth).
This version connects to AWS Bedrock AgentCore with Cognito Auth.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit cloud web application with Cognito as the Inbound Auth for the agent."""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Path to the Streamlit app
    app_path = script_dir / "ui" / "orch_web_app_cognito.py"
    
    # Check if the app file exists
    if not app_path.exists():
        print(f"Error: Could not find {app_path}")
        sys.exit(1)
    
    print("🚀 Starting E-Commerce Assistant (AWS AgentCore Version)...")
    print(f"📁 App location: {app_path}")
    print("\n" + "="*60)
    print("The application will open in your default web browser.")
    print("Press Ctrl+C to stop the server.")
    print("="*60 + "\n")
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down the application...")
        sys.exit(0)

if __name__ == "__main__":
    main()
