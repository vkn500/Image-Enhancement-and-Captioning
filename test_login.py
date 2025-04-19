import json
import os
from werkzeug.security import generate_password_hash, check_password_hash

# File paths
USERS_FILE = os.path.join('data', 'users.json')
SETTINGS_FILE = os.path.join('data', 'settings.json')

# Create a test user
test_user_id = "e61026cb-9a85-4875-a11d-e3a1c676255b"
test_username = "test"
test_email = "test@example.com"
test_password = "password123"

# Create users dictionary
users = {
    test_user_id: {
        'username': test_username,
        'email': test_email,
        'password': generate_password_hash(test_password)
    }
}

# Create user settings dictionary
user_settings = {
    test_user_id: {
        'dark_mode': False,
        'notifications': True,
        'language': 'English'
    }
}

# Save users to JSON file
try:
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)
    print(f"Successfully saved users to {USERS_FILE}")
except Exception as e:
    print(f"Error saving users data: {e}")

# Save settings to JSON file
try:
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(user_settings, f)
    print(f"Successfully saved settings to {SETTINGS_FILE}")
except Exception as e:
    print(f"Error saving settings data: {e}")

# Test loading users from JSON file
try:
    with open(USERS_FILE, 'r') as f:
        loaded_users = json.load(f)
    print(f"Successfully loaded users from {USERS_FILE}")
    print(f"Users: {loaded_users}")
except Exception as e:
    print(f"Error loading users data: {e}")

# Test authentication
if test_user_id in loaded_users:
    user_data = loaded_users[test_user_id]
    if user_data['username'] == test_username:
        if check_password_hash(user_data['password'], test_password):
            print("Authentication successful!")
        else:
            print("Authentication failed: Password does not match")
    else:
        print(f"Authentication failed: Username does not match (expected {test_username}, got {user_data['username']})")
else:
    print(f"Authentication failed: User ID {test_user_id} not found in users")

print("\nTest user credentials:")
print(f"Username: {test_username}")
print(f"Password: {test_password}")
print(f"User ID: {test_user_id}")
