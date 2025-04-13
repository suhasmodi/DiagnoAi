#!/bin/bash

echo "🚀 Starting Django project build..."

# Activate virtual environment (optional if you're using one)
if [ -d "venv" ]; then
  echo "📦 Activating virtual environment..."
  source venv/bin/activate
fi

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Export environment variables from .env (if needed)
echo "🌿 Loading environment variables from .env..."
export $(grep -v '^#' .env | xargs)

# Run migrations
echo "🛠 Applying migrations..."
python manage.py makemigrations
python manage.py migrate

# Collect static files
echo "🎨 Collecting static files..."
python manage.py collectstatic --noinput

# Optional: Run the development server
# echo "🌐 Starting development server..."
# python manage.py runserver

echo "✅ Build completed successfully!"
