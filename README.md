# IceVision Streamlit App

## Pre-requisites
You need to have the IceVision already installed in order to run the IceVision Streamlit App. You can either install the `[inference]` or  `[all]`packages. The `[inference]` option is recommended if we are only interested in getting the predictions (inference) as opposed to training models.

For `[inference]` the packages option:
```bash
pip install icevision[inference] icedata
```

For `[all]` the packages option:
```bash
pip install icevision[inference] icedata
```


## Streamlit Local Installation
From your the project root directory , run the following command:

```bash
pip install streamlit
```

---
## Running the Streamlit App from the Githup Repo

Install the streamlit package if it has not be done yet by running the following command:

```bash
streamlit run https://raw.githubusercontent.com/airctic/icevision-streamlit/master/app.py
```


## Running the Streamlit App on a Local Machine
```bash
streamlit run app.py
```
Press **Ctrl+Click** on the http://192.168.2.11:8501 URL to open the app in your local browser

---
## Docker

### Building Docker Image

You might choose your own tag name instead of `app:latest`

```bash
docker build -f Dockerfile -t ice-st:latest .
```

### Running the  Docker Image


```bash
docker run -p 8501:8501 ice-st:latest
```

> Important: Your container will be available on http://localhost:8501/

> Note: Make sure no other app is running on port 8501