# GUS data analytics Jupyter notebooks (GUSXXX_.ipynbs)

- General clean-up. Currently, they have been constantly adapted to new functionalities and most of them are in their final form, but they contain unnecessary code that is useless from the current point of view. Last .ipynb in final form: GUS01E_old_divison.ipynb.
- Move all the data related (populating the GeoTERYT Database with all the data) code form GUS02_cross_tables.ipynb to GUS01F_goeTERYT_data.ipynb and leave only cross-tables related code in GUS02_cross_tables.ipynb. 
- General change of the numbering of GUSXXX_.ipynbs - letters should be only used to indicate a next part of the same main subject. Main subjects should have separate numbers.


# GeoTERYT Database

- Change the way the numerical data for all variables are stored. Now we want to use pd.Series (instead of dict with years as keys) indexed by indexer in format 01.01.YYYY, where year comes from the dict keys. Unify all time series to time span YEAR_RANGE_FULL from geoTERYT_db.py.
- Unify the subject_ids of the census subjects, if they are overlapping. If their subject_names are equal, produce a time series from the data - with missing values between the censuses.
