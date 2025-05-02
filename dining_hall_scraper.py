import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import os
from datetime import datetime

def scrape_dining_hall(url, hall_name):
    """Scrape food names from a dining hall webpage"""
    print(f"\nScraping {hall_name}...")
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(options=options)
    try:
        print(f"Navigating to {hall_name} page...")
        driver.get(url)
        print("Waiting for page to load...")
        time.sleep(8)
        print("Extracting food items...")
        js_script = """
        function extractFoodItems() {
            let foodItems = [];
            const nonFoodKeywords = [
                'allergens', 'intolerances', 'calculator', 'available', 'preferences', 
                'add', 'contains', 'cookies', 'more', 'manager', 'meal', 'dine-in', 
                'made with', 'calories', 'select', 'save', 'manage', 'vegetarian', 
                'exclude', 'order', 'performance', 'advertising', 'closed', 'message', 
                'preview', 'hours', 'questions', 'help', 'back', 'consent', 'special', 
                'diets', 'check', 'today', 'button', 'change'
            ];
            const stationNames = [
                'the iron skillet', 'copper hood', 'green fork', 'trattoria', 
                'under the hood', 'desserts', 'breakfast', 'lunch', 'dinner',
                'salad bar', 'grill', 'pizza', 'deli', 'international', 'home zone',
                'sweets', 'beverages', 'fresh market', 'pasta', 'bakery'
            ];
            function isLikelyFood(text) {
                const lowerText = text.toLowerCase();
                if (nonFoodKeywords.some(keyword => lowerText.includes(keyword.toLowerCase()))) {
                    return false;
                }
                if (stationNames.some(station => lowerText === station.toLowerCase())) {
                    return false;
                }
                if (/\\d{1,2}:\\d{2}(am|pm|AM|PM)/.test(text) || /\\d{1,2}(am|pm|AM|PM)/.test(text)) {
                    return false;
                }
                if (/^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)/.test(text)) {
                    return false;
                }
                if (/^\\d+ Calories$/.test(text)) {
                    return false;
                }
                if (text.length < 3 || text.length > 40) {
                    return false;
                }
                return true;
            }
            const productElements = document.querySelectorAll('[data-testid="product-card-header-title"]');
            for (const elem of productElements) {
                const text = elem.textContent.trim();
                if (text && isLikelyFood(text)) {
                    foodItems.push(text);
                }
            }
            if (foodItems.length === 0) {
                const possibleFoodElements = document.querySelectorAll('h3, span:not([class*="calorie"]):not([class*="price"])');
                for (const elem of possibleFoodElements) {
                    const text = elem.textContent.trim();
                    if (!text) continue;
                    if (foodItems.includes(text)) continue;
                    if (isLikelyFood(text)) {
                        foodItems.push(text);
                    }
                }
            }
            const commonFoods = ['Pizza', 'Burger', 'Sandwich', 'Salad', 'Pasta', 'Chicken', 'Vegetable'];
            for (const foodName of commonFoods) {
                const elements = document.evaluate(
                    `//*[contains(text(), '${foodName}')]`,
                    document,
                    null,
                    XPathResult.ORDERED_NODE_SNAPSHOT_TYPE,
                    null
                );
                for (let i = 0; i < elements.snapshotLength; i++) {
                    const element = elements.snapshotItem(i);
                    const text = element.textContent.trim();
                    if (text && !foodItems.includes(text) && isLikelyFood(text)) {
                        foodItems.push(text);
                    }
                }
            }
            return foodItems;
        }
        return extractFoodItems();
        """
        food_items = driver.execute_script(js_script)
        print(f"Found {len(food_items)} food items")
        if not food_items:
            print("No food items found initially. Trying to click tabs...")
            tabs = driver.find_elements(By.CSS_SELECTOR, ".MuiTab-root, button.sc-cZYnPs, [role='tab']")
            for i, tab in enumerate(tabs):
                try:
                    tab_name = tab.text.strip() or f"Tab {i+1}"
                    print(f"Clicking on tab: {tab_name}")
                    driver.execute_script("arguments[0].scrollIntoView(true);", tab)
                    time.sleep(1)
                    driver.execute_script("arguments[0].click();", tab)
                    time.sleep(3)
                    tab_food_items = driver.execute_script(js_script)
                    print(f"Found {len(tab_food_items)} items in tab {tab_name}")
                    for item in tab_food_items:
                        if item not in food_items:
                            food_items.append(item)
                except Exception as e:
                    print(f"Error clicking tab: {e}")
        filtered_items = []
        food_keywords = ['pork', 'chicken', 'beef', 'fish', 'vegetable', 'salad', 'pasta', 'rice', 
                          'bread', 'soup', 'stew', 'burger', 'sandwich', 'wrap', 'pizza', 'quesadilla',
                          'taco', 'burrito', 'egg', 'oatmeal', 'pancake', 'waffle', 'syrup', 'toast',
                          'bagel', 'muffin', 'biscuit', 'gravy', 'potato', 'fries', 'tots', 'veggie',
                          'fruit', 'cheese', 'yogurt', 'cereal', 'granola', 'noodle', 'stir-fry',
                          'roasted', 'grilled', 'fried', 'baked', 'sautéed', 'steamed', 'sauce',
                          'bowl', 'dinner', 'lunch', 'breakfast', 'entree', 'dessert', 'pie', 'cake',
                          'cookie', 'brownie', 'ice cream', 'parfait', 'crouton', 'bean', 'lentil',
                          'bbq', 'barbecue', 'tofu', 'tempeh', 'vegan', 'gluten-free', 'organic']
        for item in food_items:
            if any(keyword in item.lower() for keyword in food_keywords):
                filtered_items.append(item)
            elif hall_name == "Observatory Hill" and item in ['Sweet Chili Pork', 'Veggie Lo Mein', 'Roasted Veggies']:
                filtered_items.append(item)
            elif len(item.split()) >= 2 and item[0].isupper():
                filtered_items.append(item)
        return filtered_items
    except Exception as e:
        print(f"Error during scraping {hall_name}: {e}")
        return []
    finally:
        driver.quit()

def main():
    """Main function to scrape all dining halls and save data"""
    print("Starting to scrape UVA dining hall menus...")
    dining_halls = {
        "Observatory Hill": "https://virginia.campusdish.com/LocationsAndMenus/ObservatoryHillDiningRoom",
        "Fresh Food Company": "https://virginia.campusdish.com/LocationsAndMenus/FreshFoodCompany",
        "Runk": "https://virginia.campusdish.com/LocationsAndMenus/Runk"
    }
    current_food_items = {}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for hall_name, url in dining_halls.items():
        food_items = scrape_dining_hall(url, hall_name)
        current_food_items[hall_name] = {
            "timestamp": timestamp,
            "items": food_items
        }
        print(f"\n{hall_name} - {len(food_items)} items found:")
        for food in food_items:
            print(f"  - {food}")
    with open("uva_dining_foods.json", 'w') as f:
        json.dump(current_food_items, f, indent=4)
    print(f"\nDining hall data updated in uva_dining_foods.json")

if __name__ == "__main__":
    main()