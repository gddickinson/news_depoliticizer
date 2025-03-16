from flask import Flask, render_template, request, jsonify
import json
from pathlib import Path
import glob
import os
from datetime import datetime

app = Flask(__name__)

# Directory where processed news is stored
PROCESSED_NEWS_DIR = "processed_news"

# Add context processor for current date/time
@app.context_processor
def inject_now():
    """Make 'now' variable available to all templates"""
    return {'now': datetime.now()}

def get_latest_news_file():
    """Get the most recently created news file"""
    files = glob.glob(f"{PROCESSED_NEWS_DIR}/depoliticized_news_*.json")
    if not files:
        return None
    
    # Sort by creation time, newest first
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def load_news_data(file_path=None):
    """Load news data from file"""
    if file_path is None:
        file_path = get_latest_news_file()
        if file_path is None:
            return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading news data: {str(e)}")
        return []

@app.route('/')
def index():
    """Main page showing the latest depoliticized news"""
    latest_file = get_latest_news_file()
    
    if latest_file:
        # Extract timestamp from filename for display
        filename = Path(latest_file).name
        timestamp = filename.replace("depoliticized_news_", "").replace(".json", "")
        formatted_time = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}"
        
        return render_template(
            'index.html', 
            news_data=load_news_data(latest_file),
            last_updated=formatted_time
        )
    else:
        return render_template('index.html', news_data=[], last_updated="No data")

@app.route('/api/news')
def api_news():
    """API endpoint for news data"""
    return jsonify(load_news_data())

@app.route('/about')
def about():
    """About page with information about the project"""
    return render_template('about.html')

# Create the templates directory and templates
def create_templates():
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Create index.html
    index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Depoliticized News</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <style>
        .news-item {
            margin-bottom: 2rem;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }
        .news-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }
        .source-tag {
            display: inline-block;
            background-color: #e9ecef;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
        }
        .rank-badge {
            background-color: #6c757d;
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-weight: bold;
            font-size: 0.85rem;
        }
        .last-updated {
            font-style: italic;
            color: #6c757d;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">Depoliticized News</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="/">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/about">About</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row mb-4">
            <div class="col-12">
                <h1>Today's News, Depoliticized</h1>
                <p class="lead">News stories ranked by importance, rewritten to remove political bias and inflammatory language.</p>
                <p class="last-updated">Last updated: {{ last_updated }}</p>
            </div>
        </div>

        <div class="row">
            {% if news_data %}
                {% for item in news_data %}
                    <div class="col-12 mb-4">
                        <div class="news-item bg-light">
                            <div class="d-flex justify-content-between align-items-start mb-3">
                                <h2>{{ item.original_title }}</h2>
                                <span class="rank-badge">Rank #{{ item.importance_rank }}</span>
                            </div>
                            
                            <div class="mb-3">
                                {% for source in item.sources %}
                                    <span class="source-tag">{{ source }}</span>
                                {% endfor %}
                            </div>
                            
                            <div class="news-content">
                                {{ item.rewritten_content|safe }}
                            </div>
                        </div>
                    </div>
                {% endfor %}
            {% else %}
                <div class="col-12">
                    <div class="alert alert-info">
                        No news data available. Please run the news processor to generate content.
                    </div>
                </div>
            {% endif %}
        </div>
    </div>

    <footer class="bg-light py-4 mt-5">
        <div class="container text-center">
            <p>Depoliticized News Aggregator &copy; {{ now.year }}</p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """
    
    # Create about.html
    about_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About - Depoliticized News</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">Depoliticized News</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="/">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link active" href="/about">About</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row">
            <div class="col-12">
                <h1>About This Project</h1>
                <p class="lead">A news aggregator that removes political bias from current events.</p>
                
                <div class="card mb-4">
                    <div class="card-header">
                        <h3>How It Works</h3>
                    </div>
                    <div class="card-body">
                        <ol>
                            <li><strong>Data Collection:</strong> News is gathered from multiple reputable sources using their official APIs.</li>
                            <li><strong>Story Matching:</strong> Similar stories across different sources are identified and grouped.</li>
                            <li><strong>Importance Ranking:</strong> Stories are ranked based on their prominence across all sources.</li>
                            <li><strong>Bias Removal:</strong> A local large language model (LLM) rewrites each story to remove political bias and inflammatory language.</li>
                            <li><strong>Presentation:</strong> The depoliticized stories are presented in order of importance.</li>
                        </ol>
                    </div>
                </div>
                
                <div class="card mb-4">
                    <div class="card-header">
                        <h3>News Sources</h3>
                    </div>
                    <div class="card-body">
                        <p>This project aggregates news from multiple sources including:</p>
                        <ul>
                            <li>NewsAPI (which provides access to hundreds of news sources)</li>
                            <li>The New York Times API</li>
                            <li>The Guardian API</li>
                        </ul>
                        <p>All sources are accessed via their official APIs in compliance with their terms of service.</p>
                    </div>
                </div>
                
                <div class="card mb-4">
                    <div class="card-header">
                        <h3>Technology</h3>
                    </div>
                    <div class="card-body">
                        <p>This project is built with:</p>
                        <ul>
                            <li><strong>Backend:</strong> Python, Flask</li>
                            <li><strong>Natural Language Processing:</strong> Local LLM (typically Llama 3 or Mistral)</li>
                            <li><strong>Frontend:</strong> HTML, CSS, Bootstrap</li>
                            <li><strong>Data Processing:</strong> scikit-learn for text similarity, numpy for data analysis</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer class="bg-light py-4 mt-5">
        <div class="container text-center">
            <p>Depoliticized News Aggregator &copy; {{ now.year }}</p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """
    
    # Write templates to files
    with open(templates_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(index_html)
    
    with open(templates_dir / "about.html", "w", encoding="utf-8") as f:
        f.write(about_html)

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run the web server for depoliticized news")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on (default: 8080)")
    args = parser.parse_args()
    
    # Create directories and templates
    Path(PROCESSED_NEWS_DIR).mkdir(exist_ok=True)
    create_templates()
    
    print(f"Starting web server on port {args.port}...")
    print(f"Open http://localhost:{args.port} in your browser to view the depoliticized news")
    
    # Run the Flask app
    app.run(debug=True, host="0.0.0.0", port=args.port)
