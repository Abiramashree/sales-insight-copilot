-- Staged layer: typed, renamed, cleaned. One row per source row.
create schema if not exists staged;

create or replace table staged.sales as
select
    cast("Store" as integer)                 as store_id,
    cast("Date" as date)                     as date,
    cast("DayOfWeek" as integer)             as dow,          -- 1 = Monday
    cast("Sales" as double)                  as sales,
    cast("Customers" as double)              as customers,
    cast("Open" as integer) = 1              as is_open,
    cast("Promo" as integer) = 1             as promo,
    nullif("StateHoliday", '0')              as state_holiday, -- a=public, b=Easter, c=Christmas
    cast("SchoolHoliday" as integer) = 1     as school_holiday
from raw.train;

create or replace table staged.stores as
select
    cast("Store" as integer)                          as store_id,
    "StoreType"                                       as store_type,
    case "Assortment" when 'a' then 'basic'
                      when 'b' then 'extra'
                      when 'c' then 'extended' end    as assortment,
    try_cast("CompetitionDistance" as double)         as competition_distance_m,
    case when try_cast("CompetitionOpenSinceYear" as integer) is not null
         then make_date(cast("CompetitionOpenSinceYear" as integer),
                        cast("CompetitionOpenSinceMonth" as integer), 1) end
                                                      as competition_open_date,
    cast("Promo2" as integer) = 1                     as promo2_enrolled,
    -- Promo2 start: Monday of ISO week `Promo2SinceWeek` in `Promo2SinceYear`
    case when try_cast("Promo2SinceYear" as integer) is not null
         then cast(date_trunc('week', make_date(cast("Promo2SinceYear" as integer), 1, 4))
                   + to_weeks(cast("Promo2SinceWeek" as integer) - 1) as date) end
                                                      as promo2_start_date,
    nullif("PromoInterval", '')                       as promo2_interval
from raw.store;
