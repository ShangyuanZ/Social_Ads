mkdir tmp_data
mkdir data

cp preliminary_contest_data/adFeature.csv data/adFeature.csv

python preparation.py
rm -r tmp_data
