import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# List of FF URLs to scrape.
def generate_urls_to_scrape_ff(start_date, end_date):
    """
    Args:
        start_date (str): The start date in the format "YYYY-MM".
        end_date (str): The end date in the format "YYYY-MM".
    
    Returns:
        list_of_urls (list): A list of URLs to scrape.
    """

    url = "https://www.forexfactory.com/calendar?month="
    list_of_urls = []
    for date in pd.date_range(start=start_date, end=end_date, freq='MS'):
        list_of_urls.append(
            f"{url}{date.strftime("%b.%Y").lower()}"
        )
    return list_of_urls

urls = generate_urls_to_scrape_ff(start_date="2024-12", end_date="2025-02")

# Mimicing browser to bypass bot detection.
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

data = []

impact_mapping = {
    "icon--ff-impact-red": "High",
    "icon--ff-impact-yel": "Medium",
    "icon--ff-impact-gry": "Low",
}

def parse_date_from_text(date_text, year):
    """
    Function to parse the date from the text, to extract the day and month from the text.
    """
    try:
        parsed = datetime.strptime(date_text + " " + year, "%b %d %Y")
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return None

# Loop over each URL in the list.
for url in urls:
    # Extract the month parameter from the URL.
    try:
        month_param = url.split("month=")[1]
        month_part, year_part = month_param.split(".")
    except Exception as e:
        print("Error extracting month and year from url:", url, e)
        continue


    # Obtain raw html -> parse it -> extract needed table.
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find("table", class_="calendar__table")
    if table is None:
        print("Table not found for url:", url)
        continue

    current_date = None
    current_time = None  

    for row in table.find_all("tr"):                # Iterate over all rows in the table, each row in table is <tr ... >
        classes = row.get("class", [])              # Getting the "..." from every row .


        # Two ways to identify new day:
        # 1. "calendar__row--day-breaker" in classes:
        if "calendar__row--day-breaker" in classes: # Check for the new day 
            td = row.find("td", class_="calendar__cell")
            if td:
                span = td.find("span")
                if span:
                    date_text = span.text.strip()  
                    new_date = parse_date_from_text(date_text, year_part)
                    if new_date:
                        current_date = new_date
                        current_time = None  
            continue  

        # 2. row.has_attr("data-day-dateline"):
        if row.has_attr("data-day-dateline"):
            try:
                timestamp = int(row["data-day-dateline"])
                event_date = datetime.fromtimestamp(timestamp)
                new_date = event_date.strftime("%Y-%m-%d")
                if new_date != current_date:
                    current_date = new_date
                    current_time = None
            except Exception:
                pass

        # Processing events (rows whose class starts with "calendar__row--" meaning that they are events)
        if any(cls.startswith("calendar__row--") for cls in classes):
            try:
                
                time_cell = row.find("td", class_="calendar__time")
                time_text = time_cell.text.strip() if time_cell else ""
                if time_text:
                    current_time = time_text
                else:
                    time_text = current_time if current_time else "N/A"

                currency = row.find("td", class_="calendar__currency").text.strip()

                # Determine impact from the impact cell.
                impact = "N/A"
                impact_td = row.find("td", class_="calendar__impact")
                if impact_td:
                    impact_span = impact_td.find("span")
                    if impact_span:
                        for cls in impact_span.get("class", []):
                            if cls in impact_mapping:
                                impact = impact_mapping[cls]
                                break

                event = row.find("td", class_="calendar__event").text.strip()
                actual = row.find("td", class_="calendar__actual").text.strip()
                forecast = row.find("td", class_="calendar__forecast").text.strip()
                previous = row.find("td", class_="calendar__previous").text.strip()

                final_date = current_date

                data.append([
                    final_date,
                    time_text,
                    currency,
                    impact,
                    event,
                    actual,
                    forecast,
                    previous
                ])
            except AttributeError:
                continue


full_df = pd.DataFrame(data, columns=["Date", "Time", "Currency", "Impact", "Event", "Actual", "Forecast", "Previous"])

# Filtering for USD and High Impact events.
filtered_df = full_df[(full_df["Currency"] == "USD") & (full_df["Impact"] == "High")].reset_index(drop=True)

filtered_df = filtered_df[(filtered_df["Actual"] != "") & 
                          (filtered_df["Forecast"] != "") & 
                          (filtered_df["Previous"] != "")]

filtered_df.reset_index(drop=True, inplace=True)

# Saving datasets.
full_df.to_csv("full_calendar_data.csv", index=False)
filtered_df.to_csv("filtered_usd_high_impact.csv", index=False)