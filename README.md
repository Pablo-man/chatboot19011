# ISO19011 AI Assistant

An interactive application that helps users learn and apply ISO19011 standards for auditing management systems. The application uses Google's Gemini AI model, LangChain, and ChromaDB to provide intelligent responses based on ISO documentation.

## Features

- **Case Study Analysis**: AI-powered analysis of ISO19011 case studies
- **Response Comparison**: Compare user responses with AI-generated solutions
- **Real-time Feedback**: Get immediate constructive feedback on your approach
- **Vector Search**: Uses semantic search to retrieve relevant information from ISO documentation
- **WebSocket Communication**: Real-time communication between client and server

## Architecture

This project integrates several modern technologies:

- **FastAPI**: High-performance web framework for building APIs
- **Google Gemini**: Large language model for generating intelligent responses
- **LangChain**: Framework for developing applications powered by language models
- **ChromaDB**: Vector database for storing and retrieving document embeddings
- **WebSockets**: For real-time bidirectional communication

## Installation

### Prerequisites

- Python 3.9 or higher
- Google Gemini API key

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/iso19011-assistant.git
   cd iso19011-assistant
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Add your ISO19011 PDF to the project root directory:
   - Name it `iso19011.pdf`
   - Ensure it's the standard document for reference

5. Create a `.env` file in the root directory with your API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-1.5-pro
   ```

## Usage

1. Start the server:
   ```
   python app.py
   ```
   Or alternatively:
   ```
   uvicorn app:app --reload
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

3. Use the interface to:
   - Submit case studies for AI analysis
   - Provide your own response to the case
   - Compare your approach with AI recommendations
   - Get feedback on areas for improvement

## API Endpoints

- `GET /`: Redirects to the web interface
- `WebSocket /init`: Main WebSocket endpoint for real-time communication

## WebSocket Communication Protocol

The application uses a simple JSON-based protocol for WebSocket communication:

### Client to Server

- New Case:
  ```json
  {
    "action": "new_case",
    "case_id": "unique_id",
    "content": "Case study text..."
  }
  ```

- Submit User Response:
  ```json
  {
    "action": "submit_user_response",
    "case_id": "unique_id",
    "content": "User's analysis..."
  }
  ```

### Server to Client

- Case Processed:
  ```json
  {
    "action": "case_processed",
    "case_id": "unique_id"
  }
  ```

- AI Response:
  ```json
  {
    "action": "ai_response",
    "case_id": "unique_id",
    "content": "AI's analysis..."
  }
  ```

- Comparison:
  ```json
  {
    "action": "comparison",
    "case_id": "unique_id",
    "content": "Comparative analysis..."
  }
  ```

## Folder Structure

```
iso19011-assistant/
├── app.py                  # Main application file
├── iso19011.pdf            # ISO standard document
├── requirements.txt        # Python dependencies
├── static/                 # Static files for the web interface
│   ├── index.html          # Main HTML file
│   ├── css/                # CSS styles
│   └── js/                 # JavaScript files
└── .env                    # Environment variables (not in git)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- ISO for providing the ISO19011 standards
- Google for the Gemini AI platform
- The LangChain and ChromaDB projects for their excellent tools