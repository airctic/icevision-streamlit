# IceVision Streamlit App


## Installation
From your the project root directory , run the following commands

```bash
pip install -r requirements.txt
```

## Usage
```bash
streamlit run app.py
```

Press **Ctrl+Click** on the http://192.168.2.11:8501 URL to open the app in your local browser

## Docker

### Building Docker Image

You might choose your own tag name instead of `app:latest`

```bash
docker build -f Dockerfile -t app:latest .
```

### Running the  Docker Image


```bash
docker run -p 8501:8501 app:latest
```

Your container will be available on http://localhost:8501/

> Note: Make sure no other app is running on port 8501