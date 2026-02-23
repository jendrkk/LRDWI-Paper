# General constant to-dos

- Adapt the .py files to changes, so that there are no bugs and current functionalities are valid.

# CBOS data

- Refactor the data extraction method in CBOS00_intro.ipynb, so that more important variables are extracted and correctly labeled for the final CBOS data frame.

# GUS data analytics Jupyter notebooks (GUSXXX_.ipynbs)

- ~~Move all the data related (populating the GeoTERYT Database with all the data) code form GUS02B_cross_tables.ipynb to GUS02A_goeTERYT_data.ipynb and leave only cross-tables related code in GUS02B_cross_tables.ipynb.~~ **DONE (v4.3)** — GUS02A now handles all data loading, processing, merging, population extraction, classification, label coding, and saving. GUS02B only builds/inspects cross tables and distributions.

## General tasks
- General clean-up. Currently, they have been constantly adapted to new functionalities and most of them are in their final form, but they contain unnecessary code that is useless from the current point of view. Last .ipynb in final form: GUS01E_old_divison.ipynb.
- General change of the numbering of GUSXXX_.ipynbs - letters should be only used to indicate a next part of the same main subject. Main subjects should have separate numbers.


# GeoTERYT Database

## Database functionalities (complete)

- ~~Fix the functionalities of children and parents. Debug the function self.link_children_to_parents(), because it throws an error.~~ **DONE (v4.2)** — `link_children_to_parents()` fixed and moved to GUS02A (runs before data loading).

## Geometries handling (complete)

- Improve memory management of the database. My idea: construct a method that will go through all the available geometries (through all geometries of the items - geoDataFrames - of the dict _geometries of the GeoTERYTDatabase) and will leave only these geometries that are unique (over all available geometries). The structure of the dict _geometries can be kept and it can still contain these geoDataFrames (one per every available year), but if one geometry is perfectly overlapping with some other we remove one of them and replace it with a "pointer" to the one that we keep.

## TERYT codes handling (complete)

- ~~Create attributes TERYTRecord.pop as pd.Series, TERYTRecord.pop_class as pd.Series and adapt the class to the changes form Data handling implementations. The series should span between 1988 and 2025 and should be indexed by date time index.~~ **DONE (v4.2)** — `TERYTRecord.pop` (pd.Series) and `TERYTRecord.pop_class` (pd.DataFrame) implemented. Population extraction via `extract_population()`, classification via `classify_population()`.

## Data handling (complete through v4.3)

- ~~Change the way the numerical data for all variables are stored. Now we want to use pd.Series (instead of dict with years as keys) indexed by indexer in format 01.01.YYYY, where year comes from the dict keys. Unify all time series to time span YEAR_RANGE_FULL from geoTERYT_db.py.~~ **DONE (v4.2)** — `DataSeries` refactored to pd.Series with DatetimeIndex (1988-2025).

- ~~Unify the subject_ids of the census subjects, if they are overlapping. If their subject_names are equal, produce a time series from the data - with missing values between the censuses.~~ **DONE (v4.3)** — Replaced destructive `unify_census_subjects()` with non-destructive `create_merged_subjects()`. Raw data preserved; new merged subjects created with 'M_' prefix. Dimensions semantically aligned across sources; range-based dimensions use common break points algorithm for unified bins. BDL data has priority; census fills only NaN years.

- ~~For the subject "P1336" we have to reduce one dimension.~~ **DONE (v4.2)** — P1336 filtering implemented in `filter_subject_data()`: keeps only "miejsce zamieszkania" (n2) and "stan na 30 czerwca" (n3).

- ~~For each teryt_id that has data from source BDL in any of the subjects with name starting with "pop__" extract the total population for every year.~~ **DONE (v4.2)** — `extract_population()` method extracts total population from "ogółem" variables, stores in `TERYTRecord.pop`. Census data included when available.

- ~~For each teryt_id that the TERYTRecord.pop data available for at least one year, we have to classify this teryt_id with respect to urban/rural split.~~ **DONE (v4.2)** — `classify_population()` implements urban/rural classification based on `rodz` and population thresholds. Stored in `TERYTRecord.pop_class` as pd.DataFrame.

- ~~Each subject_ids has dimensions n1, ..., nM and categories to translate to numerical values.~~ **DONE (v4.2)** — `code_dimension_labels()` handles sex (Mż=1, K=2), education (ordered by level), age (with lower/upper bounds extraction), household size (numeric labels get their number, non-numeric start at 101). Stored in `DataSeries.cat_code` and `DataSeries.cat_bounds`.

- Extend all data series from 1986 to 2025! Years without data should be replaced by the same structures but with np.nan values.

- Address the problem of M_hh_size that is only available for 1988, when actually it is also available for other dates. Create M_hh_size_1990 which will hold the harmonized hh_size data from census 1988 to 2002 (total range 1988 - 2002) and the variable M_hh_size_2000 which will hold harmonized hh_size data from cenus 2002 to 2021 (total range 1999 - 2025). By harmonization we mean harmonizing the variables with respect to dimension labels.

## Numerical methods (to be implemented for v5)

- to be expressed...