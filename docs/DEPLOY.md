# GitHub Pages Deployment Guide

## Quick Deploy to GitHub Pages

### Method 1: GitHub Settings (Recommended)

1. **Push your code to GitHub:**
   ```bash
   git add index.html
   git commit -m "Add landing page for GitHub Pages"
   git push origin master
   ```

2. **Enable GitHub Pages:**
   - Go to your repository: https://github.com/ihatesea69/TransUNet-Pancreas-Segmentation
   - Click **Settings** → **Pages**
   - Under "Source", select **Deploy from a branch**
   - Select branch: **master** and folder: **/ (root)**
   - Click **Save**

3. **Access your site:**
   - Your site will be live at: `https://ihatesea69.github.io/TransUNet-Pancreas-Segmentation/`
   - It may take 1-2 minutes to deploy

---

### Method 2: GitHub Actions (Automated)

Create `.github/workflows/pages.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ master ]

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Pages
        uses: actions/configure-pages@v3
      
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v2
        with:
          path: '.'
      
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v2
```

---

## Customization

### Update Links
Make sure all links point to your actual repository:
- Repository: `https://github.com/ihatesea69/TransUNet-Pancreas-Segmentation`
- Issues: Add `/issues` to repo URL
- Discussions: Add `/discussions` to repo URL

### Add Custom Domain (Optional)
1. Buy a domain (e.g., transunet.dev)
2. Add CNAME file to root:
   ```
   transunet.dev
   ```
3. Configure DNS with your domain provider

---

## Testing Locally

Before deploying, test the page locally:

```bash
# Simple HTTP server with Python
python -m http.server 8000

# Or with Node.js
npx http-server

# Then visit: http://localhost:8000
```

---

## File Structure for GitHub Pages

```
TransUNet-Pancreas-Segmentation/
├── index.html          ← Landing page (required)
├── README.md           ← Repository documentation
├── assets/             ← Images and diagrams
├── 01-04_*.ipynb      ← Notebooks
└── src/               ← Source code
```

GitHub Pages will serve `index.html` as the homepage automatically.

---

## Troubleshooting

### Page not loading?
- Check that `index.html` is in the root directory
- Verify GitHub Pages is enabled in Settings → Pages
- Clear browser cache (Ctrl+F5)

### 404 errors?
- Ensure all links use relative paths
- Check that linked files exist in the repository

### CSS not loading?
- Verify CSS is embedded in `index.html` (current setup)
- Or use relative paths if external CSS: `/style.css` not `style.css`

---

## SEO & Social Sharing

Add these meta tags to `<head>` for better social sharing:

```html
<!-- Open Graph / Facebook -->
<meta property="og:type" content="website">
<meta property="og:url" content="https://ihatesea69.github.io/TransUNet-Pancreas-Segmentation/">
<meta property="og:title" content="TransUNet: Pancreas Segmentation">
<meta property="og:description" content="AI-powered pancreas segmentation from CT scans">
<meta property="og:image" content="https://ihatesea69.github.io/TransUNet-Pancreas-Segmentation/assets/preview.png">

<!-- Twitter -->
<meta property="twitter:card" content="summary_large_image">
<meta property="twitter:url" content="https://ihatesea69.github.io/TransUNet-Pancreas-Segmentation/">
<meta property="twitter:title" content="TransUNet: Pancreas Segmentation">
<meta property="twitter:description" content="AI-powered pancreas segmentation from CT scans">
<meta property="twitter:image" content="https://ihatesea69.github.io/TransUNet-Pancreas-Segmentation/assets/preview.png">
```

---

## Next Steps

1. ✅ Create `index.html` (Done!)
2. ⬜ Push to GitHub
3. ⬜ Enable GitHub Pages in Settings
4. ⬜ Wait 1-2 minutes for deployment
5. ⬜ Visit your live site!

Your landing page will be live at:
**https://ihatesea69.github.io/TransUNet-Pancreas-Segmentation/**
