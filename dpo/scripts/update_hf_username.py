#!/usr/bin/env python3
"""
Simple script to update Hugging Face username in config for DPO models
"""

import argparse
import re

def update_hf_username(username):
    """Update HF username in config file"""
    config_path = '../../config/config.py'
    
    print(f"🔄 Updating HF username to: {username}")
    
    # Read the config file
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Replace your-username with actual username
    updated_content = content.replace('your-username/', f'{username}/')
    
    # Count replacements
    original_count = content.count('your-username/')
    updated_count = updated_content.count(f'{username}/')
    
    if original_count == 0:
        print("ℹ️  No placeholder usernames found (already updated?)")
        return
    
    # Write back the updated content
    with open(config_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✅ Updated {original_count} model entries with username: {username}")
    print("🎉 Config is ready for DPO models!")

def main():
    parser = argparse.ArgumentParser(description='Update HF username in config')
    parser.add_argument('username', help='Your Hugging Face username')
    
    args = parser.parse_args()
    
    if not args.username or args.username == 'your-username':
        print("❌ Please provide a valid Hugging Face username")
        return
    
    update_hf_username(args.username)

if __name__ == "__main__":
    main()