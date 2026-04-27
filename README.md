# E-Commerce Public Dataset Dashboard ✨

## Setup Environment - Anaconda

```bash
conda create --name main-ds python=3.10
conda activate main-ds
pip install -r requirements.txt
```

## Setup Environment - Shell/Terminal

```bash
mkdir proyek_analisis_data
cd proyek_analisis_data
pipenv install
pipenv shell
pip install -r requirements.txt
```

## Setup Environment - Virtual Environment

```bash
python -m venv venv
```

Aktifkan virtual environment:

### Windows

```bash
venv\Scripts\activate
```

### MacOS/Linux

```bash
source venv/bin/activate
```

Install library yang dibutuhkan:

```bash
pip install -r requirements.txt
```

## Run Streamlit App

Pastikan terminal berada pada folder utama project, yaitu folder `submission`.

```bash
streamlit run dashboard/dashboard.py
```

Jika command `streamlit` tidak dikenali, gunakan:

```bash
python -m streamlit run dashboard/dashboard.py
```

## Streamlit Cloud

Dashboard dapat diakses melalui link berikut:

```text
https://ecommerce-dbs-dashboard.streamlit.app/
```
