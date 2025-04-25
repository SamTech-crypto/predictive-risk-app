#!/bin/bash

set -e  # Exit on error

# Convert notebook to HTML
pip install --quiet nbconvert
jupyter nbconvert --to html predictive_risk_model.ipynb

# Set up project structure
mkdir -p notebooks
mv predictive_risk_model.html notebooks/

# Create index.html
cat << EOF > index.html
<!DOCTYPE html>
<html>
<head>
  <title>My Notebooks</title>
</head>
<body>
  <h1>Welcome to My Notebooks</h1>
  <a href="notebooks/predictive_risk_model.html">Predictive Risk Model Notebook</a>
</body>
</html>
EOF

# Create .gitignore
cat << EOF > .gitignore
*.ipynb
__pycache__/
.DS_Store
EOF

# Create netlify.toml
cat << EOF > netlify.toml
[build]
  publish = "/"
  command = ""
EOF

# Initialize Git and commit
git init
git branch -M main
git add .
git commit -m "Add notebook and Netlify config"

echo "Next steps:"
echo "1. Push to GitHub: git remote add origin <your-repo>; git push -u origin main"
echo "2. Deploy on Netlify with publish directory set to '/'"
echo "3. Test URL: https://samuelmwendwa.netlify.app/notebooks/predictive_risk_model.html"
