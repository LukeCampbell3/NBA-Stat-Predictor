import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

# NBA Team Mapping for One-Hot Encoding to Full Team Name
TEAM_ABBREVIATIONS = {
    "ATL": "Atlanta", "BOS": "Boston", "BRK": "Brooklyn", "CHI": "Chicago",
    "CHO": "Charlotte", "CLE": "Cleveland", "DAL": "Dallas", "DEN": "Denver",
    "DET": "Detroit", "GSW": "Golden State", "HOU":"Houston","IND": "Indiana", "LAC": "LA Clippers",
    "LAL": "LA Lakers", "MEM": "Memphis", "MIA": "Miami", "MIL": "Milwaukee",
    "MIN": "Minnesota", "NOP": "New Orleans", "NYK": "New York", "OKC": "Okla City",
    "ORL": "Orlando", "PHI": "Philadelphia", "PHO": "Phoenix", "POR": "Portland",
    "SAC": "Sacramento", "SAS": "San Antonio", "TOR": "Toronto", "UTA": "Utah",
    "WAS": "Washington"
}

def fetch_page(url, retry_attempts=3):
    """Fetch a webpage with retry logic."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    })

    for attempt in range(retry_attempts):
        time.sleep(2 + random.uniform(0, 1))  # Delay to avoid rate-limiting
        try:
            response = session.get(url)
            if response.status_code == 200:
                return response.content
            elif response.status_code == 429:  # Too Many Requests
                wait_time = 5
                print(f"⚠️ Rate limited (429). Waiting {wait_time} sec before retrying...")
                time.sleep(wait_time)
            else:
                print(f"❌ Failed to fetch {url} | Status Code: {response.status_code}")
                return None
        except requests.RequestException as e:
            print(f"🚨 Error fetching {url}: {e}")
            return None
    return None

def scrape_defensive_efficiency(date):
    """Scrape the last 3-game defensive efficiency rankings for the given date."""
    url = f"https://www.teamrankings.com/nba/stat/defensive-efficiency?date={date}"
    page = fetch_page(url)
    
    if not page:
        return {}

    soup = BeautifulSoup(page, 'html.parser')
    table = soup.find('table')
    if not table:
        print("⚠️ No table found on page.")
        return {}

    team_def_rtg = {}
    rows = table.find('tbody').find_all('tr')

    for row in rows:
        cols = row.find_all('td')
        if len(cols) < 5: 
            continue

        team_td = cols[1]
        team_name = team_td.attrs.get('data-sort', '').strip()        
        def_rtg_text = cols[4].attrs.get('data-sort', '').strip()  # ✅ Extracts from `data-sort`

        try:
            def_rtg = float(def_rtg_text)
        except ValueError:
            print(f"⚠️ Skipping invalid value '{def_rtg_text}' for {team_name}.")
            continue

        # Match team name to abbreviation
        for abbr, city in TEAM_ABBREVIATIONS.items():
            if city in team_name:
                team_def_rtg[abbr] = def_rtg
                break

    return team_def_rtg

def update_csv_with_def_rtg(file_path):
    """Update CSV with opponent's last 3-game defensive rating."""
    df = pd.read_csv(file_path)

    # Identify one-hot encoded opponent columns
    opp_columns = [col for col in df.columns if col.startswith('OPP_')]
    
    # Ensure oppDfRtg_3 column exists
    if 'oppDfRtg_3' not in df.columns:
        df['oppDfRtg_3'] = None  

    for index, row in df.iterrows():
        # Get game date
        game_date = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')

        # Find the opponent team abbreviation (where column value == 1)
        opp_abbr = None
        for col in opp_columns:
            if row[col] == 1:
                opp_abbr = col.replace('OPP_', '')  # Extract team abbreviation
                break  # Stop after finding the correct opponent

        if not opp_abbr:
            print(f"⚠️ No opponent found for row {index}. Skipping.")
            continue

        # Scrape defensive efficiency for the game date
        team_def_rtg = scrape_defensive_efficiency(game_date)

        # Assign the defensive rating if available
        if opp_abbr in team_def_rtg:
            df.at[index, 'oppDfRtg_3'] = team_def_rtg[opp_abbr]
        else:
            print(f"⚠️ No defensive rating found for {opp_abbr} on {game_date}.")

    # Save updated CSV
    df.to_csv(file_path, index=False)
    print(f"✅ Updated {file_path} with opponent defensive ratings.")

# Example usage
file_path = '2025.csv'  # Replace with actual file path
update_csv_with_def_rtg(file_path)
