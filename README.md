# Dataset Gathering Pipeline

### Prerequisite
-   python version `3.11.9`
-   `pip3 install -r requirements.txt`

### GUI Tool
1.  Navigate into `app` directory. (**_IMPORTANT STEP!!!_**)
2.  `streamlit run home.py`

### Structure
```
.
├── dataset_logging.csv
├── question_list.csv
├── conference_paper.csv
├── llm
│   └── gemma-3-4b-it-q4_k_m.gguf
└── app
    ├── home.py
    ├── core.py
    └── pages
```

#### Additional tools
`main.py`
-   Old pipeline version.
-   Read `question_list.csv`, `conference_paper.csv` and update `dataset_logging.csv`.

`scraper.py`
-   An `acl-anthology` api calling for scraping conference paper from different years, events, etc.
-   Give the range of years and list of venue. It will scrape 5% of each years.
-   Will update the `conference_paper.csv`.

`clustering.py`
-   An clustering technique using k-means for generating "question_type" columns.
-   Will update the `question_type.csv`.
