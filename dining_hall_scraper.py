import requests
from bs4 import BeautifulSoup
import re
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def extract_food_names(driver):
    """Extract food names from the current page"""
    food_items = []
    
    # Wait for food items to load
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='product-card-header-title']"))
        )
        
        # Find all header elements with food names
        food_elements = driver.find_elements(By.CSS_SELECTOR, "[data-testid='product-card-header-title']")
        
        for element in food_elements:
            food_name = element.text.strip()
            if food_name:
                food_items.append(food_name)
                
    except Exception as e:
        print(f"Error extracting food names: {e}")
    
    return food_items

def scrape_o_hill_menu():
    """Scrape all food names from Observatory Hill Dining Room menu"""
    url = "https://virginia.campusdish.com/LocationsAndMenus/ObservatoryHillDiningRoom"
    
    # Set up Selenium with Chrome
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    
    driver = webdriver.Chrome(options=options)
    
    all_food_items = []
    
    try:
        # Navigate to the page
        driver.get(url)
        time.sleep(5)  # Wait for page to fully load
        
        # Extract food names from the current view
        initial_foods = extract_food_names(driver)
        all_food_items.extend(initial_foods)
        
        # Check if there are different menu sections/stations
        try:
            menu_stations = driver.find_elements(By.CSS_SELECTOR, "li[role='menuitem']")
            station_count = len(menu_stations)
            
            if station_count > 1:
                # First station is already loaded, so start from index 1
                for i in range(1, station_count):
                    try:
                        # Click on each station
                        station = driver.find_elements(By.CSS_SELECTOR, "li[role='menuitem']")[i]
                        driver.execute_script("arguments[0].scrollIntoView();", station)
                        station.click()
                        time.sleep(2)  # Wait for menu items to load
                        
                        # Extract food names from this station
                        station_foods = extract_food_names(driver)
                        all_food_items.extend(station_foods)
                    except Exception as e:
                        print(f"Error clicking station {i}: {e}")
        except Exception as e:
            print(f"Error finding menu stations: {e}")
        
        # Remove duplicates
        all_food_items = list(set(all_food_items))
        
        return all_food_items
        
    except Exception as e:
        print(f"Error during scraping: {e}")
        return []
    finally:
        driver.quit()

def main():
    print("Starting to scrape UVA dining hall menus...")
    food_items = scrape_o_hill_menu()
    
    # Save to JSON
    with open("ohill_foods.json", 'w') as f:
        json.dump({"Observatory Hill": food_items}, f, indent=4)
    
    # Print results
    print(f"\nObservatory Hill - {len(food_items)} items found:")
    for food in food_items:
        print(f"  - {food}")

if __name__ == "__main__":
    main()