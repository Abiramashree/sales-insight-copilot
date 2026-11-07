-- Raw layer: land the Kaggle CSVs untouched (all varchar where types are messy).
create schema if not exists raw;

-- {until}: last day to load (the live feed replays the data a week at a time; else 9999-12-31)
create or replace table raw.train as
select * from read_csv('{raw}/train.csv', header = true, all_varchar = true)
where "Date" <= '{until}';

create or replace table raw.store as
select * from read_csv('{raw}/store.csv', header = true, all_varchar = true);
