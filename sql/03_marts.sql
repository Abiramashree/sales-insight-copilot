-- Marts: analysis-ready tables. KPI definitions live in sql/kpis.yaml.
create schema if not exists marts;

-- Stores that went dark for H2 2014 (refurbishment gap in the source data).
create or replace temp view _store_coverage as
select store_id,
       count(*)                                   as n_days,
       count(*) < (select count(distinct date) from staged.sales) as has_data_gap
from staged.sales group by 1;

create or replace table marts.dim_store as
select s.*,
       c.has_data_gap,
       coalesce(s.competition_distance_m, 100000) as competition_distance_filled_m,
       avg(f.sales) filter (where f.is_open)      as avg_daily_sales_open,
       avg(f.customers) filter (where f.is_open)  as avg_daily_customers_open
from staged.stores s
join _store_coverage c using (store_id)
join staged.sales f using (store_id)
group by all;

create or replace table marts.fct_daily_store_sales as
select
    f.store_id,
    f.date,
    f.dow,
    cast(date_trunc('week', f.date) as date)               as week_start,
    f.is_open,
    f.sales,
    f.customers,
    case when f.customers > 0 then f.sales / f.customers end as basket,
    f.promo,
    f.state_holiday,
    f.school_holiday,
    -- competition is "active" once its open date has passed (unknown date => assume active)
    s.competition_distance_m is not null
        and (s.competition_open_date is null or f.date >= s.competition_open_date)
                                                           as competition_active,
    case when s.competition_open_date is not null
         then datediff('month', s.competition_open_date, f.date) end
                                                           as months_since_competition_open,
    s.promo2_enrolled and f.date >= s.promo2_start_date    as promo2_active,
    s.promo2_enrolled and f.date >= s.promo2_start_date
        and position(strftime(f.date, '%b') in replace(s.promo2_interval, 'Sept', 'Sep')) > 0
                                                           as promo2_renewal_month
from staged.sales f
join staged.stores s using (store_id);

create or replace table marts.fct_weekly_store_sales as
select store_id,
       week_start,
       sum(sales)                                  as sales,
       sum(customers)                              as customers,
       sum(sales) / nullif(sum(customers), 0)      as basket,
       count(*) filter (where is_open)             as open_days,
       count(*) filter (where promo and is_open)   as promo_days,
       bool_or(promo2_active)                      as promo2_active
from marts.fct_daily_store_sales
group by all;
