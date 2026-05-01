#!/bin/bash
set -e
echo "🚀 Deploying Retrivo..."

cd /var/www/retrivoapp

git pull origin main

source venv/bin/activate
pip install -r requirements.txt --quiet
deactivate

sudo systemctl restart retrivoapp
echo "✅ Deployed successfully!"
sudo systemctl status retrivoapp --no-pager
