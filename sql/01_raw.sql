-- Raw layer: land the Kaggle CSVs untouched (all varchar where types are messy).
create schema if not exists raw;

create or replace table raw.train as
select * from read_csv('{raw}/train.csv', header = true, all_varchar = true);

create or replace table raw.store as
select * from read_csv('{raw}/store.csv', header = true, all_varchar = true);
