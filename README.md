# Global Air Quality Dashboard

A polished Streamlit dashboard for global air quality insights using the OpenWeather Air Pollution API. The app supports live data or deterministic demo mode, and includes a hero insights row, global snapshot, interactive map, city table, city detail analytics, forecast trends, and optional historical rolling averages.

## Highlights

- Global snapshot with best/worst city callouts
- Hero insights: global average, share of cities with poor air, cleanest city, dominant pollutant
- Interactive map with AQI legend and city snapshot panel
- City table with AQI, PM2.5/PM10, and top pollutant
- City detail view with pollutant breakdown and live observation timestamp
- Forecast and optional historical trends with rolling average
- Demo mode for quick previews without an API key (stable results per city)

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run app.py
```

## API key setup

You can provide the key in any of the following ways:

- Environment variable:

```bash
export OPENWEATHER_API_KEY="YOUR_KEY"
```

- Streamlit secrets: create `.streamlit/secrets.toml` with:

```toml
OPENWEATHER_API_KEY = "YOUR_KEY"
```

- Or paste it directly into the sidebar field.

## Sidebar controls

- `Use demo data` toggle for stable sample data
- `Select all cities` toggle and city multiselect
- `Show historical trends` toggle
- History window selector and rolling average selector

## Demo mode

If you don't have an API key yet, enable `Use demo data` in the sidebar. The app will generate stable, realistic demo data using fixed city coordinates.

## Notes

- AQI values follow OpenWeather's 1-5 scale.
- Component units are `ug/m3` as returned by the API.
- Historical trends use the OpenWeather history endpoint and display a rolling average over the selected window.
