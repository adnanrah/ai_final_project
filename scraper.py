import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import json
import os

class SimpleUVADiningScraper:
    """
    A simplified scraper for UVA dining hall menus using requests and BeautifulSoup
    """
    
    def __init__(self):
        """Initialize the scraper"""
        # Define dining locations directly
        self.dining_locations = [
            {
                'name': "Observatory Hill Dining Room",
                'url': "https://virginia.campusdish.com/LocationsAndMenus/ObservatoryHillDiningRoom",
                'short_name': "ohill"
            },
            {
                'name': "Newcomb Hall Dining Room",
                'url': "https://virginia.campusdish.com/LocationsAndMenus/NewcombHallDiningRoom",
                'short_name': "newcomb"
            },
            {
                'name': "Runk Dining Hall",
                'url': "https://virginia.campusdish.com/LocationsAndMenus/RunkDiningHall",
                'short_name': "runk"
            },
            {
                'name': "Fresh Food Company",
                'url': "https://virginia.campusdish.com/LocationsAndMenus/FreshFoodCompany",
                'short_name': "fresh"
            }
        ]
        
        # Setup headers for requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        
        # Create a session to maintain cookies
        self.session = requests.Session()
    
    def get_dining_locations(self):
        """Returns the predefined list of dining locations"""
        print(f"Using predefined list of {len(self.dining_locations)} dining locations")
        return self.dining_locations
    
    def get_menu_for_location(self, location_info):
        """
        Scrapes the menu for a specific dining location
        
        Args:
            location_info (dict): Dictionary containing location information
            
        Returns:
            list: List of food items at the location
        """
        if not location_info.get('url'):
            print(f"No URL available for {location_info.get('name', 'Unknown location')}")
            return []
        
        print(f"Getting menu for {location_info['name']} at {location_info['url']}...")
        food_items = []
        
        try:
            # Make the request
            response = self.session.get(location_info['url'], headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Print the page structure to debug
            print(f"Page title: {soup.title.string if soup.title else 'No title'}")
            
            # Look for any data objects in script tags
            script_tags = soup.find_all('script')
            menu_data = None
            
            for script in script_tags:
                script_text = script.string
                if script_text and ('window.__INITIAL_STATE__' in script_text or 'window.__PRELOADED_STATE__' in script_text):
                    print("Found data script tag")
                    try:
                        # Extract JSON data from script
                        json_str = re.search(r'window\.__(?:INITIAL|PRELOADED)_STATE__\s*=\s*({.*?});', script_text, re.DOTALL)
                        if json_str:
                            menu_data = json.loads(json_str.group(1))
                            print("Successfully extracted JSON data")
                    except Exception as e:
                        print(f"Error parsing JSON data: {e}")
            
            # If we found structured data, parse it
            if menu_data:
                return self._extract_from_json(menu_data, location_info)
            
            # Otherwise fall back to HTML parsing
            return self._extract_from_html(soup, location_info)
            
        except Exception as e:
            print(f"Error fetching menu for {location_info['name']}: {e}")
            return []
    
    def _extract_from_json(self, data, location_info):
        """Extract menu items from JSON data"""
        food_items = []
        
        try:
            # The structure might vary, so we'll try different approaches
            # We'll look for objects with food-related keys
            
            # Method 1: Look through the entire object for menu items
            if 'menu' in str(data) or 'item' in str(data) or 'product' in str(data):
                print("Found menu-related data in JSON")
                
                def extract_food_items(obj, path=""):
                    items = []
                    
                    if isinstance(obj, dict):
                        # Check if this dict looks like a food item
                        if any(key in obj for key in ['name', 'title', 'description', 'calories']):
                            if 'name' in obj or 'title' in obj:
                                name = obj.get('name', obj.get('title', ''))
                                description = obj.get('description', '')
                                calories = obj.get('calories', obj.get('nutritionalInfo', {}).get('calories'))
                                
                                if name:
                                    print(f"Found item: {name}")
                                    items.append({
                                        'name': name,
                                        'description': description,
                                        'calories': calories,
                                        'section': obj.get('category', obj.get('section', 'Menu')),
                                        'location': location_info['name'],
                                        'dining_hall': location_info['short_name']
                                    })
                        
                        # Recursively check all values
                        for key, value in obj.items():
                            items.extend(extract_food_items(value, f"{path}.{key}"))
                    
                    elif isinstance(obj, list):
                        # Check each item in the list
                        for item in obj:
                            items.extend(extract_food_items(item, path))
                    
                    return items
                
                # Extract all food items recursively
                food_items = extract_food_items(data)
                print(f"Extracted {len(food_items)} items from JSON data")
        
        except Exception as e:
            print(f"Error extracting from JSON: {e}")
        
        return food_items
    
    def _extract_from_html(self, soup, location_info):
        """Extract menu items from HTML"""
        food_items = []
        print("Falling back to HTML parsing")
        
        try:
            # Look for cards or menu item containers
            # Based on the screenshot you shared
            cards = soup.select('.card, .menu-item, [class*="menu-item"]')
            print(f"Found {len(cards)} potential menu item cards")
            
            # Also look for section headings to organize the menu
            section_headings = soup.find_all(['h2', 'h3'], class_=lambda c: c and ('section' in c or 'category' in c))
            sections = {h.text.strip(): [] for h in section_headings}
            
            # If no explicit sections found, use a default
            if not sections:
                sections = {"Menu": []}
            
            # Process each card
            for card in cards:
                try:
                    # Extract name
                    name_elem = card.select_one('h3, .card-title, .item-title, strong')
                    name = name_elem.text.strip() if name_elem else ""
                    
                    if not name:
                        # Try to find any prominent text
                        name = card.get_text().strip().split('\n')[0]
                    
                    # Extract description
                    desc_elem = card.select_one('p, .card-text, .description')
                    description = desc_elem.text.strip() if desc_elem else ""
                    
                    # Extract calories
                    calorie_elem = card.select_one('.calories, [class*="calorie"]')
                    calories = None
                    if calorie_elem:
                        calorie_text = calorie_elem.text.strip()
                        calorie_match = re.search(r'(\d+)\s*calories', calorie_text, re.IGNORECASE)
                        if calorie_match:
                            calories = int(calorie_match.group(1))
                    else:
                        # Try to find calories in the entire card text
                        card_text = card.get_text()
                        calorie_match = re.search(r'(\d+)\s*calories', card_text, re.IGNORECASE)
                        if calorie_match:
                            calories = int(calorie_match.group(1))
                    
                    # Determine which section this belongs to
                    section_name = "Menu"
                    for heading in section_headings:
                        # Check if this card is after this heading but before the next one
                        if heading.find_next(card.name, class_=card.get('class', [])) == card:
                            section_name = heading.text.strip()
                            break
                    
                    # Only add if we have a name
                    if name:
                        food_item = {
                            'name': name,
                            'description': description,
                            'calories': calories,
                            'section': section_name,
                            'location': location_info['name'],
                            'dining_hall': location_info['short_name']
                        }
                        food_items.append(food_item)
                        print(f"Added item: {name}")
                
                except Exception as e:
                    print(f"Error processing card: {e}")
            
            # If we still don't have items, try a broader approach
            if not food_items:
                print("Trying broader approach to find menu items")
                
                # Look for any elements that might contain menu items
                potential_items = []
                
                # Method 1: Look for elements with specific text patterns
                for element in soup.find_all(text=re.compile(r'\d+\s*calories', re.IGNORECASE)):
                    parent = element.parent
                    potential_items.append(parent)
                
                # Method 2: Look for elements with specific classes
                for class_hint in ['item', 'product', 'dish', 'food', 'menu']:
                    for element in soup.select(f'[class*="{class_hint}"]'):
                        potential_items.append(element)
                
                # Process potential items
                for item in potential_items:
                    try:
                        # Get all text
                        text = item.get_text().strip()
                        
                        # Skip if too short
                        if len(text) < 5:
                            continue
                        
                        # Try to extract name (first line)
                        lines = text.split('\n')
                        name = lines[0].strip()
                        
                        # Try to extract calories
                        calories = None
                        calorie_match = re.search(r'(\d+)\s*calories', text, re.IGNORECASE)
                        if calorie_match:
                            calories = int(calorie_match.group(1))
                        
                        # Try to extract description (all lines after first, except calorie line)
                        description = ' '.join([l.strip() for l in lines[1:] if 'calories' not in l.lower()])
                        
                        # Add to food items if we have a name
                        if name and name not in [i['name'] for i in food_items]:
                            food_item = {
                                'name': name,
                                'description': description,
                                'calories': calories,
                                'section': "Menu",
                                'location': location_info['name'],
                                'dining_hall': location_info['short_name']
                            }
                            food_items.append(food_item)
                            print(f"Added item using broad approach: {name}")
                    except Exception as e:
                        print(f"Error processing potential item: {e}")
        
        except Exception as e:
            print(f"Error extracting from HTML: {e}")
        
        return food_items
    
    def scrape_all(self):
        """
        Scrapes all dining locations and their menus
        
        Returns:
            pandas.DataFrame: DataFrame containing all food items
        """
        # Get all dining locations
        locations = self.get_dining_locations()
        
        # Scrape menus for each location
        all_food_items = []
        for location in locations:
            try:
                food_items = self.get_menu_for_location(location)
                all_food_items.extend(food_items)
                
                # Save what we have so far
                if food_items:
                    temp_df = pd.DataFrame(food_items)
                    temp_df.to_csv(f"uva_dining_{location['short_name']}.csv", index=False)
                    print(f"Saved {len(food_items)} items for {location['name']}")
                
                # Sleep to avoid overwhelming the server
                time.sleep(2)
            except Exception as e:
                print(f"Error scraping {location['name']}: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_food_items)
        return df
    
    def save_to_csv(self, df, filename="uva_dining_data.csv"):
        """Saves scraped data to a CSV file"""
        if df.empty:
            print("No data to save!")
            return
            
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")


def run_standalone_scraper():
    """Run the scraper as a standalone script"""
    scraper = SimpleUVADiningScraper()
    
    print("UVA Dining Hall Menu Scraper")
    print("=" * 40)
    
    # Get all locations
    locations = scraper.get_dining_locations()
    print("\nAvailable dining locations:")
    for idx, location in enumerate(locations, 1):
        print(f"{idx}. {location['name']}")
    
    while True:
        try:
            choice = input("\nWhich dining hall would you like to scrape? (1-4, 'all', or 'q' to quit): ")
            
            if choice.lower() == 'q':
                print("Exiting...")
                break
                
            if choice.lower() == 'all':
                # Scrape all locations
                print("\nScraping all dining locations...")
                df = scraper.scrape_all()
                scraper.save_to_csv(df)
                break
            
            idx = int(choice) - 1
            if 0 <= idx < len(locations):
                # Scrape single location
                location = locations[idx]
                print(f"\nScraping {location['name']}...")
                food_items = scraper.get_menu_for_location(location)
                
                if food_items:
                    df = pd.DataFrame(food_items)
                    filename = f"uva_dining_{location['short_name']}.csv"
                    df.to_csv(filename, index=False)
                    print(f"Saved {len(food_items)} items to {filename}")
                else:
                    print("No food items found!")
                
                again = input("\nWould you like to scrape another location? (y/n): ")
                if again.lower() != 'y':
                    break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number, 'all', or 'q'.")


if __name__ == "__main__":
    run_standalone_scraper()
    