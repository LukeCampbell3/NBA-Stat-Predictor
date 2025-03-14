import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import os
import time
import random  

# Persistent session to reduce bot detection
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
})

def format_player_code(full_name):
    """Generate Basketball-Reference style player code."""
    names = full_name.strip().split()
    first_name = names[0]
    last_name = names[-1]
    return last_name[:5].lower() + first_name[:2].lower() + '01'

def fetch_page(url, retry_attempts=3):
    """Fetch a webpage with delays and retry logic."""
    for attempt in range(retry_attempts):
        time.sleep(3 + random.uniform(0, 1))  # Delay BEFORE request
        try:
            response = session.get(url)
            if response.status_code == 200:
                return response.content
            elif response.status_code == 429:  # Too Many Requests
                wait_time = 3
                print(f"⚠️ Rate limited (429). Waiting {wait_time:.1f} sec before retrying...")
                time.sleep(wait_time)
            else:
                print(f"❌ Failed to fetch {url} | Status Code: {response.status_code}")
                return None
        except requests.RequestException as e:
            print(f"🚨 Error fetching {url}: {e}")
            return None
    print(f"❌ Max retries reached. Skipping {url}")
    return None

def scrape_player_game_stats(full_name, year):
    """Scrape a player's game log statistics for a given season."""
    player_code = format_player_code(full_name)
    first_letter = player_code[0]
    
    basic_url = f"https://www.basketball-reference.com/players/{first_letter}/{player_code}/gamelog/{year}"
    advanced_url = f"https://www.basketball-reference.com/players/{first_letter}/{player_code}/gamelog-advanced/{year}"
    
    basic_page = fetch_page(basic_url)
    advanced_page = fetch_page(advanced_url)
    
    if not basic_page or not advanced_page:
        return []

    basic_soup = BeautifulSoup(basic_page, 'html.parser')
    advanced_soup = BeautifulSoup(advanced_page, 'html.parser')

    basic_table = basic_soup.find('table', {'id': 'pgl_basic'})
    advanced_table = advanced_soup.find('table', {'id': 'pgl_advanced'})

    if not basic_table:
        comments = basic_soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            soup_comment = BeautifulSoup(comment, 'html.parser')
            basic_table = soup_comment.find('table', {'id': 'pgl_basic'})
            if basic_table:
                break
    
    if not advanced_table:
        comments = advanced_soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            soup_comment = BeautifulSoup(comment, 'html.parser')
            advanced_table = soup_comment.find('table', {'id': 'pgl_advanced'})
            if advanced_table:
                break

    if not basic_table or not advanced_table:
        return []

    player_stats = []
    basic_rows = basic_table.find('tbody').find_all('tr', class_=lambda x: x != 'thead')
    advanced_rows = advanced_table.find('tbody').find_all('tr', class_=lambda x: x != 'thead')

    for basic_row, advanced_row in zip(basic_rows, advanced_rows):
        game_stats = {
             'Player': full_name,
            'Date': basic_row.find('td', {'data-stat': 'date_game'}).text.strip(),
            'TM': basic_row.find('td', {'data-stat': 'team_id'}).text.strip() if basic_row.find('td', {'data-stat': 'team_id'}) else 'N/A',
            'Home/Away': basic_row.find('td', {'data-stat': 'game_location'}).text.strip() if basic_row.find('td', {'data-stat': 'game_location'}) else 'N/A',
            'OPP': advanced_row.find('td', {'data-stat': 'opp_id'}).text.strip() if advanced_row.find('td', {'data-stat': 'opp_id'}) else 'N/A',
            'MP': basic_row.find('td', {'data-stat': 'mp'}).text.strip() if basic_row.find('td', {'data-stat': 'mp'}) else 'N/A',
            'PTS': basic_row.find('td', {'data-stat': 'pts'}).text.strip() if basic_row.find('td', {'data-stat': 'pts'}) else 'N/A',
            'TRB': basic_row.find('td', {'data-stat': 'trb'}).text.strip() if basic_row.find('td', {'data-stat': 'trb'}) else 'N/A',
            'AST': basic_row.find('td', {'data-stat': 'ast'}).text.strip() if basic_row.find('td', {'data-stat': 'ast'}) else 'N/A',
            'STL': basic_row.find('td', {'data-stat': 'stl'}).text.strip() if basic_row.find('td', {'data-stat': 'stl'}) else 'N/A',
            'BLK': basic_row.find('td', {'data-stat': 'blk'}).text.strip() if basic_row.find('td', {'data-stat': 'blk'}) else 'N/A',
            'FG%': basic_row.find('td', {'data-stat': 'fg_pct'}).text.strip() if basic_row.find('td', {'data-stat': 'fg_pct'}) else 'N/A',
            '3P%': basic_row.find('td', {'data-stat': 'fg3_pct'}).text.strip() if basic_row.find('td', {'data-stat': 'fg3_pct'}) else 'N/A',
            'TS%': advanced_row.find('td', {'data-stat': 'ts_pct'}).text.strip() if advanced_row.find('td', {'data-stat': 'ts_pct'}) else 'N/A',
            'USG%': advanced_row.find('td', {'data-stat': 'usg_pct'}).text.strip() if advanced_row.find('td', {'data-stat': 'usg_pct'}) else 'N/A',
            'BPM': advanced_row.find('td', {'data-stat': 'bpm'}).text.strip() if advanced_row.find('td', {'data-stat': 'bpm'}) else 'N/A',
            'TOV': basic_row.find('td', {'data-stat': 'tov'}).text.strip() if basic_row.find('td', {'data-stat': 'tov'}) else 'N/A',
            'FT%': basic_row.find('td', {'data-stat': 'ft_pct'}).text.strip() if basic_row.find('td', {'data-stat': 'ft_pct'}) else 'N/A',
            'ORTG': advanced_row.find('td', {'data-stat': 'off_rtg'}).text.strip() if advanced_row.find('td', {'data-stat': 'off_rtg'}) else 'N/A',
            'DRTG': advanced_row.find('td', {'data-stat': 'def_rtg'}).text.strip() if advanced_row.find('td', {'data-stat': 'def_rtg'}) else 'N/A',
            'GmSc': advanced_row.find('td', {'data-stat': 'game_score'}).text.strip() if advanced_row.find('td', {'data-stat': 'game_score'}) else 'N/A',
        }
        player_stats.append(game_stats)

    return player_stats

def scrape_multiple_players(player_list, years):
    """Scrape multiple players for multiple years and save data in structured folders."""
    for player in player_list:
        player_folder = os.path.join("data", player.replace(" ", "_"))
        os.makedirs(player_folder, exist_ok=True)
        for year in years:
            stats = scrape_player_game_stats(player, str(year))
            if stats:
                df = pd.DataFrame(stats)
                csv_file = os.path.join(player_folder, f"{year}.csv")
                df.to_csv(csv_file, index=False)
                print(f"✅ Data saved for {player} ({year}) at {csv_file}")

# Example usage
players = [
  "Jaylen Brown",
  "Victor Wembanyama",
  "Anthony Davis",
  "Jalen Brunson",
  "Bam Adebayo",
  "Ja Morant",
  "Paolo Banchero",
  "Kyrie Irving",
  "Paul George",
  "Devin Booker",
  "Donovan Mitchell",
  "Jimmy Butler",
  "Damian Lillard",
  "Domantas Sabonis",
  "Tyrese Haliburton",
  "De'Aaron Fox",
  "Karl-Anthony Towns",
  "Pascal Siakam",
  "Jamal Murray",
  "Jaren Jackson Jr.",
  "Lauri Markkanen",
  "Zion Williamson",
  "Trae Young",
  "Scottie Barnes",
  "Kawhi Leonard",
  "Tyrese Maxey",
  "Julius Randle",
  "Jrue Holiday",
  "Alperen Sengun",
  "James Harden",
  "Chet Holmgren",
  "Evan Mobley",
  "Derrick White",
  "Brandon Ingram",
  "LaMelo Ball",
  "Kristaps Porzingis",
  "Bradley Beal",
  "Aaron Gordon",
  "DeMar DeRozan",
  "CJ McCollum"
]

years = [2025,2024,2023,2022,2021]
scrape_multiple_players(players, years)
