# run.py
import os
import argparse
import subprocess
import time
import webbrowser
from threading import Thread
import http.server
import socketserver
import sys

def run_scraper():
    """Run the Reddit scraper"""
    print("Running Reddit scraper...")
    from scripts.reddit_scraper import RedditScraper
    
    # Make sure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Initialize and run scraper
    scraper = RedditScraper()
    posts = scraper.scrape_all_subreddits(limit=50)  # Increase for production
    scraper.save_to_csv(posts)
    
    return True

def process_data():
    """Process the scraped data"""
    print("Processing Reddit data...")
    from models.data_processor import PCFixDataProcessor
    
    processor = PCFixDataProcessor()
    processor.process_reddit_data()
    
    return True


def run_api_server():
    """Run the Flask API server"""
    api_file = os.path.join("api", "app.py")
    subprocess.Popen([sys.executable, api_file])

def serve_frontend():
    """Serve the HTML frontend using Python's http.server"""
    # Create the directory for the frontend if it doesn't exist
    frontend_dir = os.path.join(os.getcwd(), "frontend")
    os.makedirs(frontend_dir, exist_ok=True)
    
    # Path to the HTML file
    html_path = os.path.join(frontend_dir, "index.html")
    
    # Ensure index.html exists
    if not os.path.exists(html_path):
        print(f"Error: {html_path} not found. Please create the frontend file first.")
        return False
    
    # Set up HTTP server
    os.chdir(frontend_dir)
    handler = http.server.SimpleHTTPRequestHandler
    port = 8000
    
    # Find an available port
    while True:
        try:
            httpd = socketserver.TCPServer(("", port), handler)
            break
        except OSError:
            port += 1
    
    print(f"Serving frontend at http://localhost:{port}")
    httpd.serve_forever()

def open_browser(port=8000):
    """Open browser to the frontend"""
    time.sleep(2)  # Give servers time to start
    webbrowser.open(f"http://localhost:{port}")

def main():
    parser = argparse.ArgumentParser(description="PC Fix - Reddit-based PC Troubleshooting Assistant")
    parser.add_argument("--scrape", action="store_true", help="Run the Reddit scraper")
    parser.add_argument("--process", action="store_true", help="Process scraped data")
    parser.add_argument("--all", action="store_true", help="Run the entire pipeline")
    parser.add_argument("--serve", action="store_true", help="Start the API and frontend servers")
    
    args = parser.parse_args()
    
    if args.all or args.scrape:
        run_scraper()
    
    if args.all or args.process:
        process_data()
    
    if args.all or args.serve:
        # Check if the frontend HTML file exists
        frontend_path = os.path.join("frontend", "index.html")
        if not os.path.exists(frontend_path):
            print("\nCreating frontend file...")
            with open(frontend_path, "w") as f:
                # Here you should paste the HTML content
                print("Please paste the HTML content into frontend/index.html")
        
        # Start API server in a separate thread
        api_thread = Thread(target=run_api_server)
        api_thread.daemon = True
        api_thread.start()
        
        # Open browser in a separate thread
        browser_thread = Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run frontend server in the main thread
        print("Starting frontend server...")
        serve_frontend()

if __name__ == "__main__":
    main()