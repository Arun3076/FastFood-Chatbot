import pandas as pd
import sqlite3
import random
import re
import logging
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import torch
import gradio as gr

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load fine-tuned model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("./final_model")
model = DistilBertForSequenceClassification.from_pretrained("./final_model")

# Load label encoder
df = pd.read_csv("chatbot_combined_dataset.csv")
label_encoder = LabelEncoder()
label_encoder.fit(df["Intent"].values)

# Initialize SQLite database
def init_db():
    with sqlite3.connect("fast_food.db") as conn:
        cursor = conn.cursor()
        # Create menu table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS menu (
                item_id INTEGER PRIMARY KEY,
                item_name TEXT,
                price REAL,
                category TEXT
            )
        ''')
        # Insert menu items
        menu_items = [
            ("Veg Rice", 50, "Vegetarian"),
            ("Veg Noodles", 50, "Vegetarian"),
            ("Veg Manchurian", 50, "Vegetarian"),
            ("Chicken Rice", 80, "Non-Vegetarian"),
            ("Chicken Noodles", 80, "Non-Vegetarian"),
            ("Chicken Manchurian", 80, "Non-Vegetarian")
        ]
        cursor.executemany("INSERT OR IGNORE INTO menu (item_name, price, category) VALUES (?, ?, ?)", menu_items)
        # Create orders table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER,
                item_name TEXT,
                quantity INTEGER,
                total_price REAL
            )
        ''')
        conn.commit()

# Calculate bill
def calculate_bill(order_id):
    with sqlite3.connect("fast_food.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(total_price) FROM orders WHERE order_id = ?", (order_id,))
        total = cursor.fetchone()[0]
        return total if total else 0

# Parse items from user input
def parse_items(user_input):
    menu_prices = {
        "veg rice": 50, "veg noodles": 50, "veg manchurian": 50,
        "chicken rice": 80, "chicken noodles": 80, "chicken manchurian": 80
    }
    items = []
    # Normalize input
    user_input = user_input.lower().replace("vegetarian", "veg").strip()
    # Regex to match quantity and item
    pattern = r'(\d+)\s*(veg\s*(?:rice|noodles|manchurian)|chicken\s*(?:rice|noodles|manchurian))'
    matches = re.findall(pattern, user_input)
    for quantity, item_name in matches:
        item_name = item_name.replace("  ", " ").strip()
        if item_name in menu_prices:
            quantity = int(quantity)
            total_price = quantity * menu_prices[item_name]
            item_name = " ".join(word.capitalize() for word in item_name.split())
            items.append((item_name, quantity, total_price))
    return items

# Chatbot function
def chatbot_response(user_input, order_id, chat_history):
    # Tokenize input
    encodings = tokenizer(user_input, truncation=True, padding=True, max_length=128, return_tensors="pt")
    outputs = model(**encodings)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    intent = label_encoder.inverse_transform([predicted_label])[0]
    logger.info(f"Input: {user_input}, Predicted Intent: {intent}")

    # Initialize order_id if None for relevant intents
    if order_id is None and intent in ["add item", "remove item", "confirm order", "check order status"]:
        order_id = random.randint(1000, 9999)

    # Handle intents
    if intent == "greet":
        return "", order_id, chat_history + [(user_input, "Hello! I’m Fast Food Chatbot, how can I assist you?")]
    elif intent == "menu":
        with sqlite3.connect("fast_food.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT item_name, price, category FROM menu")
            items = cursor.fetchall()
        
            # Format vegetarian items
            veg_items = [f"{item[0]} (${item[1]})" for item in items if item[2] == "Vegetarian"]
            veg_menu = "\n".join(f"• {item}" for item in veg_items) or "No vegetarian items available"
        
            # Format non-vegetarian items
            non_veg_items = [f"{item[0]} (${item[1]})" for item in items if item[2] == "Non-Vegetarian"]
            non_veg_menu = "\n".join(f"• {item}" for item in non_veg_items) or "No non-vegetarian items available"
        
            response = (
                "📋 Our Menu:\n\n"
                f"🍃 Vegetarian Options:\n{veg_menu}\n\n"
                f"🍗 Non-Vegetarian Options:\n{non_veg_menu}"
            )
        return "", order_id, chat_history + [(user_input, response)]
    # elif intent == "menu":
    #     with sqlite3.connect("fast_food.db") as conn:
    #         cursor = conn.cursor()
    #         cursor.execute("SELECT item_name, price, category FROM menu")
    #         items = cursor.fetchall()
    #         veg = [f"{item[0]} (${item[1]})" for item in items if item[2] == "Vegetarian"]
    #         non_veg = [f"{item[0]} (${item[1]})" for item in items if item[2] == "Non-Vegetarian"]
    #         response = f"Our menu: Vegetarian - {', '.join(veg)}; Non-Vegetarian - {', '.join(non_veg)}."
    #     return "", order_id, chat_history + [(user_input, response)]
    elif intent == "add item":
        items = parse_items(user_input)
        if items:
            with sqlite3.connect("fast_food.db") as conn:
                cursor = conn.cursor()
                for item_name, quantity, total_price in items:
                    cursor.execute("INSERT INTO orders (order_id, item_name, quantity, total_price) VALUES (?, ?, ?, ?)",
                                  (order_id, item_name, quantity, total_price))
                conn.commit()
            response = f"Your order of {', '.join([f'{q} {n}' for n, q, _ in items])} is placed with bill ${calculate_bill(order_id)} and your ID is {order_id}."
            return "", order_id, chat_history + [(user_input, response)]
        return "", order_id, chat_history + [(user_input, "Please specify valid items to add (e.g., 2 veg noodles).")]
    elif intent == "remove item":
        if order_id:
            with sqlite3.connect("fast_food.db") as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT item_name, quantity FROM orders WHERE order_id = ?", (order_id,))
                items = cursor.fetchall()
                removed = []
                requested_items = parse_items(user_input)
                if not requested_items:
                    return "", order_id, chat_history + [(user_input, "Please specify valid items to remove (e.g., 1 veg noodles).")]
                for item_name, quantity, _ in requested_items:
                    if (item_name.title(),) in [(i[0],) for i in items]:
                        cursor.execute("DELETE FROM orders WHERE order_id = ? AND item_name = ?", (order_id, item_name.title()))
                        removed.append(f"{quantity} {item_name.title()}")
                conn.commit()
                if removed:
                    response = f"{' and '.join(removed)} removed from order ID {order_id}. New bill: ${calculate_bill(order_id)}."
                else:
                    response = f"No matching items found in order ID {order_id}."
                return "", order_id, chat_history + [(user_input, response)]
        return "", order_id, chat_history + [(user_input, "Please provide an order ID to remove items.")]
    elif intent == "confirm order":
        if order_id:
            response = f"Order ID {order_id} confirmed. Final bill: ${calculate_bill(order_id)}."
            return "", order_id, chat_history + [(user_input, response)]
        return "", order_id, chat_history + [(user_input, "Please provide an order ID to confirm.")]
    elif intent == "check order status":
        if order_id:
            response = f"Order ID {order_id} is {'ready for pickup' if random.random() > 0.5 else 'being prepared'}."
            return "", order_id, chat_history + [(user_input, response)]
        return "", order_id, chat_history + [(user_input, "Please provide an order ID to check status.")]
    elif intent == "end greeting":
        return "", None, chat_history + [(user_input, "Goodbye! Come back soon for more tasty food!")]
    elif intent == "unrelated":
        return "", order_id, chat_history + [(user_input, "Sorry, I don’t understand you. I can assist with food orders and billing.")]
    return "", order_id, chat_history + [(user_input, "I’m not sure what you mean. Try asking about the menu or placing an order!")]

# Initialize database
init_db()

# Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("# 🍔 Fast Food Chatbot")
    order_id = gr.State(value=None)
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type your message (e.g., 'Hi', 'Show menu', 'Add 2 veg rice','Confirm my order ID {id}','Check status of order ID {id}')")
    clear = gr.Button("Clear")

    def respond(message, chat_history, order_id):
        new_message, new_order_id, new_chat_history = chatbot_response(message, order_id, chat_history)
        return new_message, new_chat_history, new_order_id

    msg.submit(respond, [msg, chatbot, order_id], [msg, chatbot, order_id])
    clear.click(lambda: (None, None), None, [chatbot, order_id], queue=False)

interface.launch()
print("Chatbot launched with Gradio and thread-safe SQLite integration!")


























# # Gradio interface
# with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as interface:
#     gr.Markdown("# 🍔 Fast Food Chatbot")
    
#     order_id = gr.State(value=None)
    
#     chatbot = gr.Chatbot(height=400)
    
#     msg = gr.Textbox(
#         placeholder="Try: 'Show menu', 'Add 2 veg rice', 'Confirm order', 'Check status'...",
#         lines=1,
#         max_lines=3
#     )
    
#     with gr.Row():
#         submit = gr.Button("Send", variant="primary", min_width=120)
#         clear = gr.Button("Clear Chat", variant="secondary", min_width=120)
#         menu_btn = gr.Button("Show Menu", variant="stop", min_width=120)

#     def respond(message, chat_history, order_id):
#         new_message, new_order_id, new_chat_history = chatbot_response(message, order_id, chat_history)
#         return new_message, new_chat_history, new_order_id

#     def show_menu(order_id, chat_history):
#         return chatbot_response("Show menu", order_id, chat_history)

#     msg.submit(respond, [msg, chatbot, order_id], [msg, chatbot, order_id])
#     submit.click(respond, [msg, chatbot, order_id], [msg, chatbot, order_id])
#     clear.click(lambda: (None, None), None, [chatbot, order_id], queue=False)
#     menu_btn.click(show_menu, [order_id, chatbot], [chatbot, order_id, msg])

# interface.launch()






# import re
# import sqlite3
# import random
# import gradio as gr
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# import torch
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from datetime import datetime
# import json
# import os

# # Constants
# DATABASE_FILE = "fast_food.db"
# MODEL_PATH = "./final_model"
# DATASET_PATH = "chatbot_combined_dataset.csv"
# CONFIG_FILE = "chatbot_config.json"

# # Initialize database
# def init_db():
#     conn = sqlite3.connect(DATABASE_FILE)
#     cursor = conn.cursor()
    
#     # Create menu table with additional columns
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS menu (
#             item_id INTEGER PRIMARY KEY,
#             item_name TEXT UNIQUE,
#             display_name TEXT,
#             price REAL,
#             category TEXT,
#             keywords TEXT,
#             description TEXT,
#             available BOOLEAN DEFAULT 1
#         )
#     ''')
    
#     # Create orders table with more details
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS orders (
#             order_id INTEGER PRIMARY KEY AUTOINCREMENT,
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#             status TEXT DEFAULT 'pending',
#             total_price REAL DEFAULT 0
#         )
#     ''')
    
#     # Create order items table
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS order_items (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             order_id INTEGER,
#             item_name TEXT,
#             quantity INTEGER,
#             price REAL,
#             FOREIGN KEY (order_id) REFERENCES orders(order_id)
#         )
#     ''')
    
#     # Check if menu is empty
#     cursor.execute("SELECT COUNT(*) FROM menu")
#     if cursor.fetchone()[0] == 0:
#         menu_items = [
#             ("veg_rice", "Veg Fried Rice", 50, "Vegetarian", "rice,veg,vegetable", "Delicious fried rice with fresh vegetables"),
#             ("veg_noodles", "Veg Hakka Noodles", 50, "Vegetarian", "noodles,noodle,veg,vegetable", "Stir-fried noodles with vegetables"),
#             ("veg_manchurian", "Veg Manchurian", 50, "Vegetarian", "manchurian,veg,vegetable", "Crispy vegetable balls in tangy sauce"),
#             ("chicken_rice", "Chicken Fried Rice", 80, "Non-Vegetarian", "rice,chicken", "Flavorful fried rice with tender chicken"),
#             ("chicken_noodles", "Chicken Hakka Noodles", 80, "Non-Vegetarian", "noodles,noodle,chicken", "Stir-fried noodles with chicken"),
#             ("chicken_manchurian", "Chicken Manchurian", 80, "Non-Vegetarian", "manchurian,chicken", "Chicken balls in tangy sauce")
#         ]
#         cursor.executemany(
#             "INSERT INTO menu (item_name, display_name, price, category, keywords, description) VALUES (?, ?, ?, ?, ?, ?)", 
#             menu_items
#         )
    
#     conn.commit()
#     conn.close()

# # Load configuration
# def load_config():
#     default_config = {
#         "responses": {
#             "greet": "Hello! I'm Fast Food Chatbot, how can I assist you today?",
#             "unrelated": "I'm sorry, I didn't understand that. I can help with menu, orders, and order status.",
#             "fallback": "I'm not sure I follow. Could you rephrase that?"
#         },
#         "order_statuses": ["preparing", "ready", "delivered"],
#         "theme": {
#             "primary_color": "#FF6B6B",
#             "secondary_color": "#4ECDC4"
#         }
#     }
    
#     if os.path.exists(CONFIG_FILE):
#         with open(CONFIG_FILE, 'r') as f:
#             try:
#                 return {**default_config, **json.load(f)}
#             except json.JSONDecodeError:
#                 return default_config
#     return default_config

# config = load_config()

# # Database connection decorator
# def with_db_connection(func):
#     def wrapper(*args, **kwargs):
#         conn = sqlite3.connect(DATABASE_FILE)
#         try:
#             result = func(conn, *args, **kwargs)
#             conn.commit()
#             return result
#         finally:
#             conn.close()
#     return wrapper

# # Load model and tokenizer
# @with_db_connection
# def load_model(conn):
#     tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
#     model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    
#     # Load label encoder
#     df = pd.read_csv(DATASET_PATH)
#     label_encoder = LabelEncoder()
#     label_encoder.fit(df["Intent"].values)
    
#     return tokenizer, model, label_encoder

# tokenizer, model, label_encoder = load_model()

# # Enhanced order parsing
# @with_db_connection
# def parse_order_items(conn, user_input):
#     cursor = conn.cursor()
#     cursor.execute("SELECT item_name, display_name, keywords FROM menu WHERE available = 1")
#     menu_items = cursor.fetchall()
    
#     item_map = {}
#     synonym_map = {}
#     for item_name, display_name, keywords in menu_items:
#         for keyword in keywords.split(','):
#             key = keyword.strip().lower()
#             item_map[key] = (item_name, display_name)
#             # Create number variations (1 chicken -> chicken, 2 chickens -> chicken)
#             if key.endswith('s'):
#                 singular = key[:-1]
#                 item_map[singular] = (item_name, display_name)
#                 synonym_map[singular] = key
    
#     # Improved regex pattern with better quantity handling
#     pattern = r'(\d+)?\s*(.+?)(?:\s|$|,|\.)'
#     matches = re.findall(pattern, user_input.lower())
    
#     order_items = []
#     for quantity, item_desc in matches:
#         quantity = int(quantity) if quantity else 1
        
#         # Clean item description
#         item_desc = item_desc.strip()
#         if not item_desc:
#             continue
            
#         # Check for direct matches first
#         direct_match = item_map.get(item_desc)
#         if direct_match:
#             order_items.append((*direct_match, quantity))
#             continue
            
#         # Check for partial matches
#         best_match = None
#         best_score = 0
        
#         for keyword, (item_name, display_name) in item_map.items():
#             if keyword in item_desc:
#                 score = len(keyword)
#                 if score > best_score:
#                     best_score = score
#                     best_match = (item_name, display_name, quantity)
        
#         if best_match:
#             order_items.append(best_match)
    
#     return order_items

# # Order management functions
# @with_db_connection
# def create_order(conn, items):
#     cursor = conn.cursor()
#     total = sum(price * quantity for _, _, price, quantity in items)
    
#     cursor.execute(
#         "INSERT INTO orders (total_price) VALUES (?)",
#         (total,)
#     )
#     order_id = cursor.lastrowid
    
#     for item_name, display_name, price, quantity in items:
#         cursor.execute(
#             "INSERT INTO order_items (order_id, item_name, quantity, price) VALUES (?, ?, ?, ?)",
#             (order_id, item_name, quantity, price)
#         )
    
#     return order_id, total

# @with_db_connection
# def update_order(conn, order_id, items_to_add=None, items_to_remove=None):
#     cursor = conn.cursor()
    
#     if items_to_remove:
#         for item_name in items_to_remove:
#             cursor.execute(
#                 "DELETE FROM order_items WHERE order_id = ? AND item_name = ?",
#                 (order_id, item_name)
#             )
    
#     if items_to_add:
#         for item_name, display_name, price, quantity in items_to_add:
#             cursor.execute(
#                 """INSERT INTO order_items (order_id, item_name, quantity, price) 
#                    VALUES (?, ?, ?, ?)
#                    ON CONFLICT(order_id, item_name) 
#                    DO UPDATE SET quantity = quantity + excluded.quantity""",
#                 (order_id, item_name, quantity, price)
#             )
    
#     # Recalculate total
#     cursor.execute(
#         "SELECT SUM(price * quantity) FROM order_items WHERE order_id = ?",
#         (order_id,)
#     )
#     total = cursor.fetchone()[0] or 0
    
#     cursor.execute(
#         "UPDATE orders SET total_price = ? WHERE order_id = ?",
#         (total, order_id)
#     )
    
#     return get_order_details(conn, order_id)

# @with_db_connection
# def get_order_details(conn, order_id):
#     cursor = conn.cursor()
    
#     cursor.execute(
#         """SELECT m.display_name, oi.quantity, oi.price 
#            FROM order_items oi
#            JOIN menu m ON oi.item_name = m.item_name
#            WHERE oi.order_id = ?""",
#         (order_id,)
#     )
#     items = cursor.fetchall()
    
#     cursor.execute(
#         "SELECT total_price, status, created_at FROM orders WHERE order_id = ?",
#         (order_id,)
#     )
#     total, status, created_at = cursor.fetchone()
    
#     return {
#         "order_id": order_id,
#         "items": items,
#         "total": total,
#         "status": status,
#         "created_at": created_at
#     }

# # Chatbot response generation
# def generate_response(intent, user_input, order_id=None):
#     if intent == "greet":
#         return config["responses"]["greet"]
    
#     elif intent == "menu":
#         return generate_menu_response()
    
#     elif intent == "add item":
#         return handle_add_item(user_input, order_id)
    
#     elif intent == "remove item":
#         return handle_remove_item(user_input, order_id)
    
#     elif intent == "confirm order":
#         return handle_confirm_order(order_id)
    
#     elif intent == "check order status":
#         return handle_check_status(order_id)
    
#     elif intent == "end greeting":
#         return "Goodbye! Thank you for visiting our fast food restaurant!"
    
#     elif intent == "unrelated":
#         return config["responses"]["unrelated"]
    
#     return config["responses"]["fallback"]

# @with_db_connection
# def generate_menu_response(conn):
#     cursor = conn.cursor()
#     cursor.execute("""
#         SELECT display_name, price, category, description 
#         FROM menu 
#         WHERE available = 1
#         ORDER BY category, display_name
#     """)
#     items = cursor.fetchall()
    
#     menu_by_category = {}
#     for name, price, category, desc in items:
#         if category not in menu_by_category:
#             menu_by_category[category] = []
#         menu_by_category[category].append(f"**{name}** - ${price}\n_{desc}_")
    
#     response = "**Our Menu**\n\n"
#     for category, items in menu_by_category.items():
#         response += f"**{category.upper()}**\n"
#         response += "\n".join(f"- {item}" for item in items)
#         response += "\n\n"
    
#     return response

# def handle_add_item(user_input, order_id):
#     order_items = parse_order_items(user_input)
#     if not order_items:
#         return "Please specify items to add (e.g., '2 veg noodles, 1 chicken rice')"
    
#     # Get prices for items
#     items_with_prices = []
#     with sqlite3.connect(DATABASE_FILE) as conn:
#         cursor = conn.cursor()
#         for item_name, display_name, quantity in order_items:
#             cursor.execute(
#                 "SELECT price FROM menu WHERE item_name = ?",
#                 (item_name,)
#             )
#             price = cursor.fetchone()[0]
#             items_with_prices.append((item_name, display_name, price, quantity))
    
#     if order_id:
#         order_details = update_order(
#             order_id,
#             items_to_add=items_with_prices
#         )
#         response = format_order_response(order_details, "Order updated")
#     else:
#         order_id, total = create_order(items_with_prices)
#         order_details = get_order_details(order_id)
#         response = format_order_response(order_details, "New order created")
    
#     return response

# def handle_remove_item(user_input, order_id):
#     if not order_id:
#         return "Please provide an order ID to remove items."
    
#     order_items = parse_order_items(user_input)
#     if not order_items:
#         return "Please specify which items to remove (e.g., 'remove 1 veg noodles')"
    
#     items_to_remove = [item_name for item_name, _, _ in order_items]
#     order_details = update_order(
#         order_id,
#         items_to_remove=items_to_remove
#     )
    
#     if not order_details["items"]:
#         return f"Order #{order_id} is now empty."
    
#     return format_order_response(order_details, "Items removed from order")

# def handle_confirm_order(order_id):
#     if not order_id:
#         return "Please provide an order ID to confirm."
    
#     order_details = get_order_details(order_id)
#     if not order_details["items"]:
#         return "Your order is empty. Please add items first."
    
#     # Update status to confirmed
#     with sqlite3.connect(DATABASE_FILE) as conn:
#         cursor = conn.cursor()
#         cursor.execute(
#             "UPDATE orders SET status = 'confirmed' WHERE order_id = ?",
#             (order_id,)
#         )
#         conn.commit()
    
#     order_details = get_order_details(order_id)
#     response = format_order_response(order_details, "Order confirmed!")
#     response += "\n\nThank you for your order! Your food will be ready soon."
#     return response

# def handle_check_status(order_id):
#     if not order_id:
#         return "Please provide an order ID to check status."
    
#     order_details = get_order_details(order_id)
#     if not order_details:
#         return f"Order #{order_id} not found."
    
#     status_messages = {
#         "pending": "is being prepared",
#         "confirmed": "is being prepared",
#         "ready": "is ready for pickup!",
#         "delivered": "has been delivered"
#     }
    
#     status_msg = status_messages.get(
#         order_details["status"],
#         f"has status: {order_details['status']}"
#     )
    
#     response = f"Order #{order_id} {status_msg}\n"
#     response += f"\nCreated at: {order_details['created_at']}\n"
#     response += f"\nTotal: ${order_details['total']:.2f}"
    
#     return response

# def format_order_response(order_details, title):
#     response = f"**{title} #{order_details['order_id']}**\n"
#     response += "\n".join(
#         f"- {q} × {name} (${p:.2f})" 
#         for name, q, p in order_details["items"]
#     )
#     response += f"\n\n**Total: ${order_details['total']:.2f}**"
#     return response

# # Chatbot function with context
# def chatbot_response(user_input, chat_history, order_id=None):
#     # Tokenize input
#     encodings = tokenizer(user_input, truncation=True, padding=True, max_length=128, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**encodings)
#     predicted_label = torch.argmax(outputs.logits, dim=1).item()
#     intent = label_encoder.inverse_transform([predicted_label])[0]
    
#     # Generate response based on intent
#     response = generate_response(intent, user_input, order_id)
    
#     # Update chat history
#     chat_history.append((user_input, response))
    
#     return chat_history, order_id if order_id else ""

# # Gradio Interface
# def create_chat_interface():
#     with gr.Blocks(theme=gr.themes.Default(
#         primary_hue="orange",
#         secondary_hue="teal"
#     )) as demo:
#         gr.Markdown("# 🍔 Fast Food Order Chatbot")
#         gr.Markdown("Welcome to our fast food restaurant! How can I help you today?")
        
#         with gr.Row():
#             with gr.Column(scale=3):
#                 chatbot = gr.Chatbot(
#                     label="Chat History",
#                     height=500,
#                     bubble_full_width=False
#                 )
                
#                 with gr.Row():
#                     user_input = gr.Textbox(
#                         placeholder="Type your message...",
#                         show_label=False,
#                         scale=4
#                     )
#                     submit_btn = gr.Button("Send", variant="primary")
                
#                 order_id = gr.Textbox(
#                     label="Order ID (if applicable)",
#                     placeholder="Leave empty for new orders"
#                 )
                
#                 clear_btn = gr.Button("Clear Chat")
            
#             with gr.Column(scale=1):
#                 gr.Markdown("### Quick Actions")
#                 gr.Examples(
#                     examples=[
#                         "Show me the menu",
#                         "I'd like 2 chicken rice and 1 veg noodles",
#                         "What's the status of order #123?",
#                         "Remove 1 chicken rice from my order"
#                     ],
#                     inputs=user_input,
#                     label="Try these examples"
#                 )
                
#                 with gr.Accordion("View Full Menu", open=False):
#                     menu_display = gr.Markdown()
                
#                 with gr.Accordion("Order Management", open=False):
#                     with gr.Group():
#                         new_order_btn = gr.Button("Start New Order")
#                         check_status_btn = gr.Button("Check Order Status")
#                         confirm_order_btn = gr.Button("Confirm Order")
        
#         # Event handlers
#         submit_btn.click(
#             fn=chatbot_response,
#             inputs=[user_input, chatbot, order_id],
#             outputs=[chatbot, order_id]
#         ).then(lambda: "", outputs=user_input)
        
#         user_input.submit(
#             fn=chatbot_response,
#             inputs=[user_input, chatbot, order_id],
#             outputs=[chatbot, order_id]
#         ).then(lambda: "", outputs=user_input)
        
#         clear_btn.click(
#             fn=lambda: ([], ""),
#             outputs=[chatbot, order_id]
#         )
        
#         new_order_btn.click(
#             fn=lambda: ([], str(random.randint(1000, 9999))),
#             outputs=[chatbot, order_id]
#         )
        
#         check_status_btn.click(
#             fn=lambda chat, oid: (
#                 chat + [("Check order status", handle_check_status(oid))],
#                 oid
#             ),
#             inputs=[chatbot, order_id],
#             outputs=[chatbot, order_id]
#         )
        
#         confirm_order_btn.click(
#             fn=lambda chat, oid: (
#                 chat + [("Confirm order", handle_confirm_order(oid))],
#                 oid
#             ),
#             inputs=[chatbot, order_id],
#             outputs=[chatbot, order_id]
#         )
        
#         demo.load(
#             fn=generate_menu_response,
#             outputs=menu_display
#         )
    
#     return demo

# # Initialize and run
# if __name__ == "__main__":
#     init_db()
#     demo = create_chat_interface()
#     demo.launch()

# old interface

# import pandas as pd
# import sqlite3
# import random
# import re
# import logging
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# from sklearn.preprocessing import LabelEncoder
# import torch
# import gradio as gr

# # Set up logging for debugging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load fine-tuned model and tokenizer
# tokenizer = DistilBertTokenizer.from_pretrained("./final_model")
# model = DistilBertForSequenceClassification.from_pretrained("./final_model")

# # Load label encoder
# df = pd.read_csv("chatbot_combined_dataset.csv")
# label_encoder = LabelEncoder()
# label_encoder.fit(df["Intent"].values)

# # Initialize SQLite database
# def init_db():
#     with sqlite3.connect("fast_food.db") as conn:
#         cursor = conn.cursor()
#         # Create menu table
#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS menu (
#                 item_id INTEGER PRIMARY KEY,
#                 item_name TEXT,
#                 price REAL,
#                 category TEXT
#             )
#         ''')
#         # Insert menu items
#         menu_items = [
#             ("Veg Rice", 50, "Vegetarian"),
#             ("Veg Noodles", 50, "Vegetarian"),
#             ("Veg Manchurian", 50, "Vegetarian"),
#             ("Chicken Rice", 80, "Non-Vegetarian"),
#             ("Chicken Noodles", 80, "Non-Vegetarian"),
#             ("Chicken Manchurian", 80, "Non-Vegetarian")
#         ]
#         cursor.executemany("INSERT OR IGNORE INTO menu (item_name, price, category) VALUES (?, ?, ?)", menu_items)
#         # Create orders table
#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS orders (
#                 order_id INTEGER,
#                 item_name TEXT,
#                 quantity INTEGER,
#                 total_price REAL
#             )
#         ''')
#         conn.commit()

# # Calculate bill
# def calculate_bill(order_id):
#     with sqlite3.connect("fast_food.db") as conn:
#         cursor = conn.cursor()
#         cursor.execute("SELECT SUM(total_price) FROM orders WHERE order_id = ?", (order_id,))
#         total = cursor.fetchone()[0]
#         return total if total else 0

# # Parse items from user input
# def parse_items(user_input):
#     menu_prices = {
#         "veg rice": 50, "veg noodles": 50, "veg manchurian": 50,
#         "chicken rice": 80, "chicken noodles": 80, "chicken manchurian": 80
#     }
#     items = []
#     # Normalize input
#     user_input = user_input.lower().replace("vegetarian", "veg").strip()
#     # Regex to match quantity and item
#     pattern = r'(\d+)\s*(veg\s*(?:rice|noodles|manchurian)|chicken\s*(?:rice|noodles|manchurian))'
#     matches = re.findall(pattern, user_input)
#     for quantity, item_name in matches:
#         item_name = item_name.replace("  ", " ").strip()
#         if item_name in menu_prices:
#             quantity = int(quantity)
#             total_price = quantity * menu_prices[item_name]
#             item_name = " ".join(word.capitalize() for word in item_name.split())
#             items.append((item_name, quantity, total_price))
#     return items

# # Chatbot function
# def chatbot_response(user_input, order_id, chat_history):
#     # Tokenize input
#     encodings = tokenizer(user_input, truncation=True, padding=True, max_length=128, return_tensors="pt")
#     outputs = model(**encodings)
#     predicted_label = torch.argmax(outputs.logits, dim=1).item()
#     intent = label_encoder.inverse_transform([predicted_label])[0]
#     logger.info(f"Input: {user_input}, Predicted Intent: {intent}")

#     # Initialize order_id if None for relevant intents
#     if order_id is None and intent in ["add item", "remove item", "confirm order", "check order status"]:
#         order_id = random.randint(1000, 9999)

#     # Handle intents
#     if intent == "greet":
#         return "", order_id, chat_history + [(user_input, "Hello! I’m Fast Food Chatbot, how can I assist you?")]
#     elif intent == "menu":
#         with sqlite3.connect("fast_food.db") as conn:
#             cursor = conn.cursor()
#             cursor.execute("SELECT item_name, price, category FROM menu")
#             items = cursor.fetchall()
#             veg = [f"{item[0]} (${item[1]})" for item in items if item[2] == "Vegetarian"]
#             non_veg = [f"{item[0]} (${item[1]})" for item in items if item[2] == "Non-Vegetarian"]
#             response = f"Our menu: Vegetarian - {', '.join(veg)}; Non-Vegetarian - {', '.join(non_veg)}."
#         return "", order_id, chat_history + [(user_input, response)]
#     elif intent == "add item":
#         items = parse_items(user_input)
#         if items:
#             with sqlite3.connect("fast_food.db") as conn:
#                 cursor = conn.cursor()
#                 for item_name, quantity, total_price in items:
#                     cursor.execute("INSERT INTO orders (order_id, item_name, quantity, total_price) VALUES (?, ?, ?, ?)",
#                                   (order_id, item_name, quantity, total_price))
#                 conn.commit()
#             response = f"Your order of {', '.join([f'{q} {n}' for n, q, _ in items])} is placed with bill ${calculate_bill(order_id)} and your ID is {order_id}."
#             return "", order_id, chat_history + [(user_input, response)]
#         return "", order_id, chat_history + [(user_input, "Please specify valid items to add (e.g., 2 veg noodles).")]
#     elif intent == "remove item":
#         if order_id:
#             with sqlite3.connect("fast_food.db") as conn:
#                 cursor = conn.cursor()
#                 cursor.execute("SELECT item_name, quantity FROM orders WHERE order_id = ?", (order_id,))
#                 items = cursor.fetchall()
#                 removed = []
#                 requested_items = parse_items(user_input)
#                 if not requested_items:
#                     return "", order_id, chat_history + [(user_input, "Please specify valid items to remove (e.g., 1 veg noodles).")]
#                 for item_name, quantity, _ in requested_items:
#                     if (item_name.title(),) in [(i[0],) for i in items]:
#                         cursor.execute("DELETE FROM orders WHERE order_id = ? AND item_name = ?", (order_id, item_name.title()))
#                         removed.append(f"{quantity} {item_name.title()}")
#                 conn.commit()
#                 if removed:
#                     response = f"{' and '.join(removed)} removed from order ID {order_id}. New bill: ${calculate_bill(order_id)}."
#                 else:
#                     response = f"No matching items found in order ID {order_id}."
#                 return "", order_id, chat_history + [(user_input, response)]
#         return "", order_id, chat_history + [(user_input, "Please provide an order ID to remove items.")]
#     elif intent == "confirm order":
#         if order_id:
#             response = f"Order ID {order_id} confirmed. Final bill: ${calculate_bill(order_id)}."
#             return "", order_id, chat_history + [(user_input, response)]
#         return "", order_id, chat_history + [(user_input, "Please provide an order ID to confirm.")]
#     elif intent == "check order status":
#         if order_id:
#             response = f"Order ID {order_id} is {'ready for pickup' if random.random() > 0.5 else 'being prepared'}."
#             return "", order_id, chat_history + [(user_input, response)]
#         return "", order_id, chat_history + [(user_input, "Please provide an order ID to check status.")]
#     elif intent == "end greeting":
#         return "", None, chat_history + [(user_input, "Goodbye! Come back soon for more tasty food!")]
#     elif intent == "unrelated":
#         return "", order_id, chat_history + [(user_input, "Sorry, I don’t understand you. I can assist with food orders and billing.")]
#     return "", order_id, chat_history + [(user_input, "I’m not sure what you mean. Try asking about the menu or placing an order!")]

# # Initialize database
# init_db()

# # Gradio interface
# with gr.Blocks() as interface:
#     gr.Markdown("# Fast Food Chatbot")
#     order_id = gr.State(value=None)
#     chatbot = gr.Chatbot()
#     msg = gr.Textbox(placeholder="Type your message (e.g., 'Hi', 'Show menu', 'Add 2 veg noodles')")
#     clear = gr.Button("Clear")

#     def respond(message, chat_history, order_id):
#         new_message, new_order_id, new_chat_history = chatbot_response(message, order_id, chat_history)
#         return new_message, new_chat_history, new_order_id

#     msg.submit(respond, [msg, chatbot, order_id], [msg, chatbot, order_id])
#     clear.click(lambda: (None, None), None, [chatbot, order_id], queue=False)

# interface.launch()
# print("Chatbot launched with Gradio and thread-safe SQLite integration!")



# import re
# import pandas as pd
# import sqlite3
# import random
# import streamlit as st
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# from sklearn.preprocessing import LabelEncoder
# import torch

# # Initialize session state
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'order_id' not in st.session_state:
#     st.session_state.order_id = None

# # Database functions
# def get_db_connection():
#     """Create a new database connection for each thread"""
#     return sqlite3.connect("fast_food.db", check_same_thread=False)

# def init_db():
#     """Initialize database tables"""
#     with get_db_connection() as conn:
#         cursor = conn.cursor()
        
#         # Create menu table with additional columns
#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS menu (
#                 item_id INTEGER PRIMARY KEY,
#                 item_name TEXT UNIQUE,
#                 display_name TEXT,
#                 price REAL,
#                 category TEXT,
#                 keywords TEXT
#             )
#         ''')
        
#         # Check if menu is empty
#         cursor.execute("SELECT COUNT(*) FROM menu")
#         if cursor.fetchone()[0] == 0:
#             menu_items = [
#                 ("veg_rice", "Veg Rice", 50, "Vegetarian", "rice,veg,vegetable"),
#                 ("veg_noodles", "Veg Noodles", 50, "Vegetarian", "noodles,noodle,veg,vegetable"),
#                 ("veg_manchurian", "Veg Manchurian", 50, "Vegetarian", "manchurian,veg,vegetable"),
#                 ("chicken_rice", "Chicken Rice", 80, "Non-Vegetarian", "rice,chicken"),
#                 ("chicken_noodles", "Chicken Noodles", 80, "Non-Vegetarian", "noodles,noodle,chicken"),
#                 ("chicken_manchurian", "Chicken Manchurian", 80, "Non-Vegetarian", "manchurian,chicken")
#             ]
#             cursor.executemany(
#                 "INSERT INTO menu (item_name, display_name, price, category, keywords) VALUES (?, ?, ?, ?, ?)", 
#                 menu_items
#             )
        
#         # Create orders table
#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS orders (
#                 order_id INTEGER,
#                 item_name TEXT,
#                 quantity INTEGER,
#                 total_price REAL,
#                 PRIMARY KEY (order_id, item_name)
#             )
#         ''')
#         conn.commit()

# # Initialize database at startup
# init_db()

# # Load model and tokenizer
# @st.cache_resource
# def load_model():
#     tokenizer = DistilBertTokenizer.from_pretrained("./final_model")
#     model = DistilBertForSequenceClassification.from_pretrained("./final_model")
#     return tokenizer, model

# tokenizer, model = load_model()

# # Load label encoder
# @st.cache_data
# def load_labels():
#     df = pd.read_csv("chatbot_combined_dataset.csv")
#     label_encoder = LabelEncoder()
#     label_encoder.fit(df["Intent"].values)
#     return label_encoder

# label_encoder = load_labels()

# # Function to calculate bill
# def calculate_bill(order_id):
#     with get_db_connection() as conn:
#         cursor = conn.cursor()
#         cursor.execute("""
#             SELECT m.display_name, o.quantity, o.total_price 
#             FROM orders o
#             JOIN menu m ON o.item_name = m.item_name
#             WHERE o.order_id = ?
#         """, (order_id,))
#         items = cursor.fetchall()
        
#         cursor.execute("SELECT SUM(total_price) FROM orders WHERE order_id = ?", (order_id,))
#         total = cursor.fetchone()[0] or 0
        
#         return items, total

# # Enhanced order parsing
# def parse_order_items(user_input):
#     with get_db_connection() as conn:
#         cursor = conn.cursor()
#         cursor.execute("SELECT item_name, display_name, keywords FROM menu")
#         menu_items = cursor.fetchall()
    
#     # Create mapping of keywords to menu items
#     item_map = {}
#     for item_id, display_name, keywords in menu_items:
#         for keyword in keywords.split(','):
#             item_map[keyword.strip()] = (item_id, display_name)
    
#     # Regex pattern to match quantities and items
#     pattern = r'(\d+)\s*(.+?)(?:\s|$|,)'
#     matches = re.findall(pattern, user_input.lower())
    
#     order_items = []
#     for quantity, item_desc in matches:
#         # Find best matching menu item
#         best_match = None
#         best_score = 0
        
#         for keyword, (item_id, display_name) in item_map.items():
#             if keyword in item_desc:
#                 # Score based on keyword length (longer matches are better)
#                 score = len(keyword)
#                 if score > best_score:
#                     best_score = score
#                     best_match = (item_id, display_name, int(quantity))
        
#         if best_match:
#             order_items.append(best_match)
    
#     return order_items

# # Chatbot function
# def chatbot_response(user_input, order_id=None):
#     # Tokenize input
#     encodings = tokenizer(user_input, truncation=True, padding=True, max_length=128, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**encodings)
#     predicted_label = torch.argmax(outputs.logits, dim=1).item()
#     intent = label_encoder.inverse_transform([predicted_label])[0]

#     # Handle intents
#     if intent == "greet":
#         return "Hello! I'm Fast Food Chatbot, how can I assist you?"
#     elif intent == "menu":
#         with get_db_connection() as conn:
#             cursor = conn.cursor()
#             cursor.execute("SELECT display_name, price, category FROM menu")
#             items = cursor.fetchall()
#             veg = [f"{item[0]} (${item[1]})" for item in items if item[2] == "Vegetarian"]
#             non_veg = [f"{item[0]} (${item[1]})" for item in items if item[2] == "Non-Vegetarian"]
#             return f"Our menu:\n\nVegetarian:\n- " + "\n- ".join(veg) + "\n\nNon-Vegetarian:\n- " + "\n- ".join(non_veg)
#     elif intent == "add item":
#         order_items = parse_order_items(user_input)
        
#         if order_items:
#             order_id = order_id or random.randint(1000, 9999)
#             st.session_state.order_id = order_id
            
#             with get_db_connection() as conn:
#                 cursor = conn.cursor()
#                 for item_id, display_name, quantity in order_items:
#                     cursor.execute("SELECT price FROM menu WHERE item_name = ?", (item_id,))
#                     price = cursor.fetchone()[0]
#                     total_price = price * quantity
                    
#                     cursor.execute("""
#                         INSERT OR REPLACE INTO orders (order_id, item_name, quantity, total_price) 
#                         VALUES (?, ?, ?, ?)
#                     """, (order_id, item_id, quantity, total_price))
#                 conn.commit()
            
#             items, total = calculate_bill(order_id)
#             response = f"Order #{order_id} updated:\n"
#             response += "\n".join([f"- {q} × {name} (${price})" for name, q, price in items])
#             response += f"\n\nTotal: ${total}"
#             return response
        
#         return "Please specify items to add (e.g., '2 veg noodles, 1 chicken rice')"
#     elif intent == "remove item":
#         if order_id:
#             order_items = parse_order_items(user_input)
            
#             if order_items:
#                 with get_db_connection() as conn:
#                     cursor = conn.cursor()
#                     for item_id, _, _ in order_items:
#                         cursor.execute("DELETE FROM orders WHERE order_id = ? AND item_name = ?", 
#                                      (order_id, item_id))
#                     conn.commit()
                
#                 items, total = calculate_bill(order_id)
#                 if items:
#                     response = f"Order #{order_id} updated:\n"
#                     response += "\n".join([f"- {q} × {name} (${price})" for name, q, price in items])
#                     response += f"\n\nTotal: ${total}"
#                 else:
#                     response = f"Order #{order_id} is now empty"
#                 return response
            
#             return "Please specify which items to remove (e.g., 'remove 1 veg noodles')"
#         return "Please provide an order ID to remove items."
#     elif intent == "confirm order":
#         if order_id:
#             items, total = calculate_bill(order_id)
#             if items:
#                 response = f"Order #{order_id} confirmed!\n\n"
#                 response += "\n".join([f"- {q} × {name}" for name, q, _ in items])
#                 response += f"\n\nTotal: ${total}\n\nThank you for your order!"
#                 return response
#             return "Your order is empty. Please add items first."
#         return "Please provide an order ID to confirm."
#     elif intent == "check order status":
#         if order_id:
#             status = "ready for pickup" if random.random() > 0.5 else "being prepared"
#             return f"Order #{order_id} status:\n\n{status.upper()}"
#         return "Please provide an order ID to check status."
#     elif intent == "end greeting":
#         return "Goodbye! Come back soon for more tasty food!"
#     elif intent == "unrelated":
#         return "Sorry, I don't understand. I can help with:\n- Showing menu\n- Taking orders\n- Checking order status"
#     return "I'm not sure what you mean. Try asking about our menu or placing an order!"

# # Streamlit UI
# st.title("🍔 Fast Food Chatbot")

# # Chat container
# chat_container = st.container()
# with chat_container:
#     for message in st.session_state.chat_history:
#         with st.chat_message("user"):
#             st.write(message[0])
#         with st.chat_message("assistant"):
#             st.write(message[1])

# # Input area
# col1, col2 = st.columns([4, 1])
# with col1:
#     user_input = st.chat_input("Type your message...")
# with col2:
#     if st.button("Clear Chat"):
#         st.session_state.chat_history = []
#         st.rerun()

# # Order ID input
# order_id_input = st.text_input("Order ID (if needed)", 
#                               value=str(st.session_state.order_id) if st.session_state.order_id else "")

# # Process input
# if user_input:
#     # Add user message to chat
#     st.session_state.chat_history.append((user_input, ""))
    
#     # Get response
#     try:
#         order_id = int(order_id_input) if order_id_input else None
#     except ValueError:
#         order_id = None
    
#     response = chatbot_response(user_input, order_id)
    
#     # Update chat history
#     st.session_state.chat_history[-1] = (user_input, response)
#     st.rerun()





























# import pandas as pd
# import sqlite3
# import random
# import streamlit as st
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# from sklearn.preprocessing import LabelEncoder
# import torch

# # Initialize session state
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'order_id' not in st.session_state:
#     st.session_state.order_id = None

# # Database functions
# def get_db_connection():
#     """Create a new database connection for each thread"""
#     conn = sqlite3.connect("fast_food.db", check_same_thread=False)
#     return conn

# def init_db():
#     """Initialize database tables"""
#     with get_db_connection() as conn:
#         cursor = conn.cursor()
        
#         # Create menu table
#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS menu (
#                 item_id INTEGER PRIMARY KEY,
#                 item_name TEXT,
#                 price REAL,
#                 category TEXT
#             )
#         ''')
        
#         # Check if menu is empty
#         cursor.execute("SELECT COUNT(*) FROM menu")
#         if cursor.fetchone()[0] == 0:
#             menu_items = [
#                 ("Veg Rice", 50, "Vegetarian"),
#                 ("Veg Noodles", 50, "Vegetarian"),
#                 ("Veg Manchurian", 50, "Vegetarian"),
#                 ("Chicken Rice", 80, "Non-Vegetarian"),
#                 ("Chicken Noodles", 80, "Non-Vegetarian"),
#                 ("Chicken Manchurian", 80, "Non-Vegetarian")
#             ]
#             cursor.executemany("INSERT INTO menu (item_name, price, category) VALUES (?, ?, ?)", menu_items)
        
#         # Create orders table
#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS orders (
#                 order_id INTEGER,
#                 item_name TEXT,
#                 quantity INTEGER,
#                 total_price REAL,
#                 PRIMARY KEY (order_id, item_name)
#             )
#         ''')
#         conn.commit()

# # Initialize database at startup
# init_db()

# # Load model and tokenizer
# @st.cache_resource
# def load_model():
#     tokenizer = DistilBertTokenizer.from_pretrained("./final_model")
#     model = DistilBertForSequenceClassification.from_pretrained("./final_model")
#     return tokenizer, model

# tokenizer, model = load_model()

# # Load label encoder
# @st.cache_data
# def load_labels():
#     df = pd.read_csv("chatbot_combined_dataset.csv")
#     label_encoder = LabelEncoder()
#     label_encoder.fit(df["Intent"].values)
#     return label_encoder

# label_encoder = load_labels()

# # Function to calculate bill
# def calculate_bill(order_id):
#     with get_db_connection() as conn:
#         cursor = conn.cursor()
#         cursor.execute("SELECT SUM(total_price) FROM orders WHERE order_id = ?", (order_id,))
#         total = cursor.fetchone()[0]
#         return total if total else 0

# # Chatbot function
# def chatbot_response(user_input, order_id=None):
#     # Tokenize input
#     encodings = tokenizer(user_input, truncation=True, padding=True, max_length=128, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**encodings)
#     predicted_label = torch.argmax(outputs.logits, dim=1).item()
#     intent = label_encoder.inverse_transform([predicted_label])[0]

#     # Handle intents
#     if intent == "greet":
#         return "Hello! I'm Fast Food Chatbot, how can I assist you?"
#     elif intent == "menu":
#         with get_db_connection() as conn:
#             cursor = conn.cursor()
#             cursor.execute("SELECT item_name, price, category FROM menu")
#             items = cursor.fetchall()
#             veg = [f"{item[0]} (${item[1]})" for item in items if item[2] == "Vegetarian"]
#             non_veg = [f"{item[0]} (${item[1]})" for item in items if item[2] == "Non-Vegetarian"]
#             return f"Our menu:\n\nVegetarian:\n- " + "\n- ".join(veg) + "\n\nNon-Vegetarian:\n- " + "\n- ".join(non_veg)
#     elif intent == "add item":
#         items = []
#         if "veg noodles" in user_input.lower():
#             items.append(("Veg Noodles", 2, 50 * 2))
#         elif "chicken rice" in user_input.lower():
#             items.append(("Chicken Rice", 1, 80 * 1))
#         elif "veg rice" in user_input.lower() and "chicken noodles" in user_input.lower():
#             items.append(("Veg Rice", 1, 50 * 1))
#             items.append(("Chicken Noodles", 1, 80 * 1))
        
#         if items:
#             order_id = order_id or random.randint(1000, 9999)
#             st.session_state.order_id = order_id
#             with get_db_connection() as conn:
#                 cursor = conn.cursor()
#                 for item_name, quantity, total_price in items:
#                     cursor.execute("""
#                         INSERT OR REPLACE INTO orders (order_id, item_name, quantity, total_price) 
#                         VALUES (?, ?, ?, ?)
#                     """, (order_id, item_name, quantity, total_price))
#                 conn.commit()
#             return f"Added to order #{order_id}:\n" + "\n".join([f"- {q} {n}" for n, q, _ in items]) + f"\n\nCurrent total: ${calculate_bill(order_id)}"
#         return "Please specify items to add (e.g., '2 veg noodles')"
#     elif intent == "remove item":
#         if order_id:
#             with get_db_connection() as conn:
#                 cursor = conn.cursor()
#                 if "veg noodles" in user_input.lower():
#                     cursor.execute("DELETE FROM orders WHERE order_id = ? AND item_name = ?", (order_id, "Veg Noodles"))
#                 elif "chicken rice" in user_input.lower():
#                     cursor.execute("DELETE FROM orders WHERE order_id = ? AND item_name = ?", (order_id, "Chicken Rice"))
#                 conn.commit()
#             return f"Items removed from order #{order_id}\nNew total: ${calculate_bill(order_id)}"
#         return "Please provide an order ID to remove items."
#     elif intent == "confirm order":
#         if order_id:
#             return f"Order #{order_id} confirmed!\n\nFinal amount: ${calculate_bill(order_id)}\n\nThank you for your order!"
#         return "Please provide an order ID to confirm."
#     elif intent == "check order status":
#         if order_id:
#             status = "ready for pickup" if random.random() > 0.5 else "being prepared"
#             return f"Order #{order_id} status:\n\n{status.upper()}"
#         return "Please provide an order ID to check status."
#     elif intent == "end greeting":
#         return "Goodbye! Come back soon for more tasty food!"
#     elif intent == "unrelated":
#         return "Sorry, I don't understand. I can help with:\n- Showing menu\n- Taking orders\n- Checking order status"
#     return "I'm not sure what you mean. Try asking about our menu or placing an order!"

# # Streamlit UI
# st.title("🍔 Fast Food Chatbot")

# # Chat container
# chat_container = st.container()
# with chat_container:
#     for message in st.session_state.chat_history:
#         with st.chat_message("user"):
#             st.write(message[0])
#         with st.chat_message("assistant"):
#             st.write(message[1])
# # Input area
# col1, col2 = st.columns([4, 1])
# with col1:
#     user_input = st.chat_input("Type your message...")
# with col2:
#     if st.button("Clear Chat"):
#         st.session_state.chat_history = []
#         st.rerun()

# # Order ID input
# order_id_input = st.text_input("Order ID (if needed)", value=str(st.session_state.order_id) if st.session_state.order_id else "")

# # Process input
# if user_input:
#     # Add user message to chat
#     st.session_state.chat_history.append((user_input, ""))
    
#     # Get response
#     try:
#         order_id = int(order_id_input) if order_id_input else None
#     except ValueError:
#         order_id = None
#     response = chatbot_response(user_input, order_id)
    
#     # Update chat history
#     st.session_state.chat_history[-1] = (user_input, response)
#     st.rerun()