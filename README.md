# Global Air Quality Dashboard

A Streamlit dashboard that pulls air quality data from the OpenWeather Air Pollution API and visualizes global city analytics.

## Quick start

1. Create an OpenWeather API key.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

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

## Notes

- AQI values follow OpenWeather's 1-5 scale.
- Component units are `ug/m3` as returned by the API.
- Historical trends use the OpenWeather history endpoint and display a rolling average over the selected window.
