# News Depoliticizer

A tool that aggregates news from multiple sources, ranks stories by importance, and uses a local LLM to rewrite them without political bias or inflammatory language.

## Overview

News Depoliticizer helps readers consume news without the political spin and inflammatory language that often accompanies today's reporting. The project:

1. Legally collects news from multiple APIs (no web scraping)
2. Identifies and groups similar stories across sources
3. Ranks stories by prominence/importance
4. Uses a local LLM (via Ollama) to rewrite each story in a neutral, balanced tone
5. Presents the depoliticized stories through a clean web interface

## Features

- **Multi-source aggregation**: Collects news from NewsAPI, NY Times API, and The Guardian API
- **Story grouping**: Identifies the same stories across different sources
- **Importance ranking**: Orders stories based on their prominence across news outlets
- **Bias removal**: Rewrites stories to remove political bias and inflammatory language
- **Local processing**: All LLM operations run on your local machine, ensuring privacy
- **Clean web interface**: Mobile-friendly display of depoliticized news stories

## Requirements

- Python 3.8+
- Ollama (with llama3 model installed)
- API keys for news sources (optional, sample data provided)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/news-depoliticizer.git
cd news-depoliticizer
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install required packages

```bash
pip install requests pandas numpy scikit-learn python-dotenv flask
```

### 4. Install Ollama (if not already installed)

Follow the [Ollama installation instructions](https://ollama.ai/download) for your platform.

Make sure the `llama3` model is installed:

```bash
ollama pull llama3
```

### 5. Set up API keys (optional)

Create a `.env` file in the project directory with your API keys:

```
NEWSAPI_KEY=your_newsapi_key
NYT_KEY=your_nyt_key
GUARDIAN_KEY=your_guardian_key
```

You can get free API keys from:
- [NewsAPI](https://newsapi.org/)
- [New York Times API](https://developer.nytimes.com/)
- [The Guardian API](https://open-platform.theguardian.com/)

## Usage

### Quick Start (Full Pipeline)

Run the complete pipeline with one command:

```bash
python run_depoliticizer.py
```

This will:
1. Fetch news from APIs (or use sample data if no API keys)
2. Process news with your local LLM
3. Start a web server at http://localhost:8080

### Running Components Individually

#### 1. Fetch news

```bash
python news_aggregator.py
```

#### 2. Process with LLM

```bash
python llm_processor.py
```

#### 3. Start web server

```bash
python web_display.py
```

### Command Line Options

The runner script supports several options:

```bash
python run_depoliticizer.py --help
```

Common options:
- `--ollama-model MODEL`: Use a different Ollama model (default: llama3)
- `--skip-fetch`: Skip news fetching, use existing data
- `--skip-process`: Skip LLM processing, use existing processed data
- `--web-only`: Only start the web server
- `--port PORT`: Run the web server on a specific port (default: 8080)

## Architecture

The project consists of three main components:

### 1. News Aggregator (`news_aggregator.py`)

- Fetches articles from multiple news APIs
- Groups similar stories using TF-IDF and cosine similarity
- Ranks stories by their prominence across sources
- Prepares data for LLM processing

### 2. LLM Processor (`llm_processor.py`)

- Interfaces with local LLM (Ollama)
- Instructs the LLM to rewrite articles in a neutral tone
- Post-processes LLM output to clean up artifacts
- Saves processed news data for display

### 3. Web Interface (`web_display.py`)

- Serves a Flask web application
- Displays depoliticized news in a clean, mobile-friendly interface
- Shows articles in ranked order with source attribution

### 4. Runner Script (`run_depoliticizer.py`)

- Coordinates the entire pipeline
- Provides command-line options for customization
- Handles error checking and dependency verification

## File Structure

```
news-depoliticizer/
├── news_aggregator.py      # News fetching and grouping
├── llm_processor.py        # LLM rewriting engine
├── web_display.py          # Web interface
├── run_depoliticizer.py    # Runner script
├── test_ollama.py          # Ollama integration test
├── .env                    # API keys (create this file)
├── processed_news/         # Folder for processed articles
├── templates/              # Web templates
│   ├── index.html          # Homepage template
│   └── about.html          # About page template
└── README.md               # This file
```

## Ethical Considerations

### Attribution

This tool respects the original news sources by:
- Only using official APIs (no scraping)
- Clearly attributing sources in the interface
- Maintaining the factual integrity of the original reporting

### Neutrality vs. False Equivalence

The aim of depoliticizing news is not to create false equivalence between viewpoints but to:
- Remove inflammatory language
- Present facts in a neutral tone
- Include relevant perspectives when appropriate
- Allow readers to form their own opinions

## Known Limitations

- Limited to news available through partner APIs
- LLM rewrites may occasionally miss subtle political framing
- Processing time can be slow depending on your hardware
- Without API keys, only sample/demo data is available

## Troubleshooting

### Common Issues

1. **Port conflicts**: If port 8080 is in use, specify an alternative:
   ```bash
   python web_display.py --port 9000
   ```

2. **Ollama errors**: Ensure Ollama is running and the llama3 model is installed:
   ```bash
   ollama list
   ollama pull llama3
   ```

3. **Missing API data**: If you don't have API keys, the system will use sample data.

4. **Slow processing**: LLM processing speed depends on your hardware. Be patient, or try a smaller model.

## Future Improvements

- Support for more news sources and languages
- Advanced bias detection algorithms
- User customization of neutrality parameters
- RSS feed and email digest options
- Improved story grouping algorithm
- Docker containerization for easier deployment

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Ollama](https://ollama.ai/) for local LLM inference
- News data provided by [NewsAPI](https://newsapi.org/), [NY Times API](https://developer.nytimes.com/), and [The Guardian API](https://open-platform.theguardian.com/)
- Inspired by the need for less polarized news consumption
