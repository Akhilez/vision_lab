def yearly_to_hourly(yearly: int) -> float:
    weekly = yearly / 52
    daily = weekly / 5
    hourly = daily / 8

    return hourly


def hourly_to_yearly(hourly: float) -> float:
    daily = hourly * 8
    weekly = daily * 5
    yearly = weekly * 52

    return yearly


yearly = 120 * 1000
hourly = yearly_to_hourly(yearly)
print(yearly, hourly)


hourly = 58
yearly = hourly_to_yearly(hourly)
print(yearly, hourly)


yearly = 150 * 1000
hourly = yearly_to_hourly(yearly)
print(yearly, hourly)

hourly = 65
yearly = hourly_to_yearly(hourly)
print(yearly, hourly)

hourly = 50
yearly = hourly_to_yearly(hourly)
print(yearly, hourly)
