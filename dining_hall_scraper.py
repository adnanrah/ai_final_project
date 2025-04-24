import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

def scrape_o_hill_menu():
    """Scrape all food names from Observatory Hill Dining Room menu"""
    url = "https://virginia.campusdish.com/LocationsAndMenus/ObservatoryHillDiningRoom"
    
    # Set up Selenium with Chrome
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    
    driver = webdriver.Chrome(options=options)
    
    try:
        print("Navigating to the dining hall page...")
        driver.get(url)
        
        print("Waiting for page to load...")
        time.sleep(8)
        
        print("Extracting food items...")
        
        # Use JavaScript to extract only actual menu items
        js_script = """
        function extractFoodItems() {
            let foodItems = [];
            
            // Non-food keywords to filter out
            const nonFoodKeywords = [
                'allergens', 'intolerances', 'calculator', 'available', 'preferences', 
                'add', 'contains', 'cookies', 'more', 'manager', 'meal', 'dine-in', 
                'made with', 'calories', 'select', 'save', 'manage', 'vegetarian', 
                'exclude', 'order', 'performance', 'advertising', 'closed', 'message', 
                'preview', 'hours', 'questions', 'help', 'back', 'consent', 'special', 
                'diets', 'check', 'today', 'button', 'change'
            ];
            
            // Food stations to filter out
            const stationNames = [
                'the iron skillet', 'copper hood', 'green fork', 'trattoria', 
                'under the hood', 'desserts', 'breakfast', 'lunch', 'dinner'
            ];
            
            // Function to check if an item is likely a food
            function isLikelyFood(text) {
                // Convert to lowercase for comparison
                const lowerText = text.toLowerCase();
                
                // Check if it contains any non-food keywords
                if (nonFoodKeywords.some(keyword => lowerText.includes(keyword.toLowerCase()))) {
                    return false;
                }
                
                // Check if it's a station name
                if (stationNames.some(station => lowerText === station.toLowerCase())) {
                    return false;
                }
                
                // Check if it contains time patterns (like opening hours)
                if (/\\d{1,2}:\\d{2}(am|pm|AM|PM)/.test(text) || /\\d{1,2}(am|pm|AM|PM)/.test(text)) {
                    return false;
                }
                
                // Filter out items that are just days of the week
                if (/^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)/.test(text)) {
                    return false;
                }
                
                // Filter out items that are just numbers (like calorie counts)
                if (/^\\d+ Calories$/.test(text)) {
                    return false;
                }
                
                // Check length - most food names are between 3 and 40 characters
                if (text.length < 3 || text.length > 40) {
                    return false;
                }
                
                return true;
            }
            
            // Try to find food items in specific elements first
            const productElements = document.querySelectorAll('[data-testid="product-card-header-title"]');
            for (const elem of productElements) {
                const text = elem.textContent.trim();
                if (text && isLikelyFood(text)) {
                    foodItems.push(text);
                }
            }
            
            // If we didn't find any food items with specific selectors, try a more general approach
            if (foodItems.length === 0) {
                // Look for elements that might contain food names
                const possibleFoodElements = document.querySelectorAll('h3, span:not([class*="calorie"]):not([class*="price"])');
                
                for (const elem of possibleFoodElements) {
                    const text = elem.textContent.trim();
                    
                    // Skip empty text
                    if (!text) continue;
                    
                    // Skip if it's already in our list
                    if (foodItems.includes(text)) continue;
                    
                    // Check if it's likely a food item
                    if (isLikelyFood(text)) {
                        foodItems.push(text);
                    }
                }
            }
            
            // Final check - look for items from your screenshots
            const specificFoods = ['Sweet Chili Pork', 'Veggie Lo Mein', 'Roasted Veggies'];
            for (const foodName of specificFoods) {
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
        
        # Extract food items using JavaScript
        food_items = driver.execute_script(js_script)
        print(f"Found {len(food_items)} food items")
        
        # If we still don't have any food items, try clicking on tabs
        if not food_items:
            print("No food items found initially. Trying to click tabs...")
            
            # Try to find and click on station tabs
            tabs = driver.find_elements(By.CSS_SELECTOR, ".MuiTab-root, button.sc-cZYnPs, [role='tab']")
            
            for i, tab in enumerate(tabs):
                try:
                    tab_name = tab.text.strip() or f"Tab {i+1}"
                    print(f"Clicking on tab: {tab_name}")
                    
                    # Scroll to and click the tab
                    driver.execute_script("arguments[0].scrollIntoView(true);", tab)
                    time.sleep(1)
                    driver.execute_script("arguments[0].click();", tab)
                    time.sleep(3)
                    
                    # Extract food items again
                    tab_food_items = driver.execute_script(js_script)
                    print(f"Found {len(tab_food_items)} items in tab {tab_name}")
                    
                    # Add to our list
                    for item in tab_food_items:
                        if item not in food_items:
                            food_items.append(item)
                except Exception as e:
                    print(f"Error clicking tab: {e}")
        
        # Post-processing to further filter non-food items
        filtered_items = []
        food_keywords = ['pork', 'chicken', 'beef', 'fish', 'vegetable', 'salad', 'pasta', 'rice', 
                          'bread', 'soup', 'stew', 'burger', 'sandwich', 'wrap', 'pizza', 'quesadilla',
                          'taco', 'burrito', 'egg', 'oatmeal', 'pancake', 'waffle', 'syrup', 'toast',
                          'bagel', 'muffin', 'biscuit', 'gravy', 'potato', 'fries', 'tots', 'veggie',
                          'fruit', 'cheese', 'yogurt', 'cereal', 'granola', 'noodle', 'stir-fry',
                          'roasted', 'grilled', 'fried', 'baked', 'sautéed', 'steamed', 'sauce']
        
        for item in food_items:
            # Check if the item contains any food keywords
            if any(keyword in item.lower() for keyword in food_keywords):
                filtered_items.append(item)
            # Also include items from your screenshots as they're confirmed food items
            elif item in ['Sweet Chili Pork', 'Veggie Lo Mein', 'Roasted Veggies']:
                filtered_items.append(item)
        
        return filtered_items
        
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