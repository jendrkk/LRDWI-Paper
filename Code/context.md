# Survey

We will analyze the answers of respondents of two (country-level) weighted surveys: CBOS Akturalne Problemy i Wydarzenia (1990-2017) and GUS Household Budget Survey (1986, 1992, 1995, 1999, 2004-2023). The main variable of interest is the household income. We have full demographic specification of the respondent and the size of household. We know the region from which the respondent comes from and in which type of locality/human settelment/town they come from. We want to conduct a deep and plausible regional income inequality analysis.

# GeoTERYT Database

The whole goal for this database is to create cross tables age vs sex, sex vs education and tables household size for years 1986 - 2024 on the lowest territorial level possible (gmina is the lowest). These tables show the structure and distribution of certain variables in the local populations. Information from the cross tables will be used to regionally re-weight survey data, which is already weighted on the country level. The survey data has answers to questions about age, gender, highest education, household size, city size (or if respondent lives in rural areas). There is also indication in which voivodship (till 2000: old voivodships, from 1999: new voivodships) lives the respondent.

1. The idea:
    - We divide the survey answers into subgroups of voivodships and assign weight to create a representative sample.
    - Since in each voivodships there is only one city which is greater than 500.000 inhabitants (and we know the respondents that have chosen this answer), we can extract only these observations and (since we have GeoTERYT database) re-weight this subsample to be representative for this exact city.

2. What geoTERYT has to provide:
    - Cross tables sex vs age, sex vs education and tables of household size.
        - Very precise for new voivodships for years 1999-2024. These is a lot of data here and we do not have to use predictions.
        - Precise for old voivodships for years 1986-2000. Here we have much less data and we have to join data from gimnas that are assigned to old voivodships.
        - Precise for all gminas for 1986-2024. This data will allow us both to create the data for old voivodships and to do the analysis of cities or groups of cities of a similar size (along with the categories of cities in the survey: rural area, cities <20_000, cities 20_000-50_000, cities 50_000-100_000, cities 100_000-500_000, cities >500_000).
    - In ideal case: for every teryt_id we have precise cross tables and tables of the aforementioned variables for the years 1986 - 2023 (we have survey data only from these years).

3. The problem:
    
    Time series inside the objects DataSeries contain missing data - there is no data for some years. The cross tables sex vs age group, sex vs education and the tables household size are therefore full with np.nan for some years or for some territorial units.

4. Reliable data that we have:
    - ALL gminas that we present from 1986 till 2024.
    - We know which gminas where in which old voivodships.
    - Sex groups for gminas 1988
    - Age groups for gminas 1988
    - Education groups for gminas 1988
    - Sex vs age for old voivodships 1986-1994
    - Sex vs educ for whole country 1986-88 & 1991-94
    - Sex vs age for gminas 1995-2024
    - Sex vs educ for gminas 2002, 2021
    - Sex vs educ for powiaty (collection of gimnas or single big cities - this is important for us) 2011
    - Education groups for new voivodships 1995-2024 (+ warsaw separated from mazovian voi. and mazovian voi. without warsaw for 2000+)
    - Household w.r.t. their sizes (1,2,3 and 4, 5+) for gminas 1988
    - Household w.r.t. their sizes (1,2,3,4,5+) for gminas 2002, 2021
    - Household w.r.t. their sizes (1,2,3,4,5+) for powiaty 2011

5. Numerical Solutions:
    We have to prepare the database for implementation of a state-of-art numerical algorithm/method that will allow us to predict and "interpolate" the cross tables and tables produced from M_ subjects that will describe the (joint) distributions of some characteristics for territorial units of the lowest level. We will do two separate predictions/interpolations inside two prediction sections:
    - For the years 1986 - 2002. We will name it Prediction1990.
        - We use Censuses 1988 and 2002 as main anchors.
    - For the years 1999 - 2024. We will name it Prediction2000.
        - We use Censuses 2002, 2011 and 2021 as main anchors.
    
    In order to be able to design any numerical solution for our problem, we have to unify the labels for groups of each variable inside each prediction sections.