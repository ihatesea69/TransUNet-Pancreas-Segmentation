# Quick Fix: Missing IEEEtran.cls

## Problem
```
! LaTeX Error: File `IEEEtran.cls' not found.
```

## Solution (Choose one method)

### Method 1: MiKTeX Console (EASIEST)

1. **Open MiKTeX Console** (search in Start menu)
2. Click **"Check for updates"** button
3. Go to **"Packages"** tab
4. In search box, type: `ieeetran`
5. Find **"ieeetran"** package, click **"+"** button to install
6. Also install these packages:
   - `cite`
   - `amsmath`
   - `graphicx`
   - `booktabs`
   - `hyperref`
7. Click **"Apply"** or **"Install"**

### Method 2: Command Line (FAST)

```powershell
# Install IEEEtran package
mpm --install=ieeetran

# Install other required packages
mpm --install=cite
mpm --install=amsmath
mpm --install=graphicx
mpm --install=booktabs
mpm --install=hyperref
mpm --install=url
mpm --install=multirow
mpm --install=algorithm2e

# Refresh database
initexmf --update-fndb
```

### Method 3: Enable Auto-Install (AUTOMATIC)

1. Open **MiKTeX Console**
2. Go to **"Settings"**
3. Set **"Install missing packages"** to **"Yes"** or **"Ask me first"**
4. Close MiKTeX Console
5. Run compile script again - it will auto-install missing packages

## After Installation

Run compile script again:
```powershell
.\compile.ps1
```

## Troubleshooting

### If mpm command not found:
```powershell
# Add MiKTeX to PATH
$env:Path += ";C:\Program Files\MiKTeX\miktex\bin\x64"
```

### If packages still not found:
```powershell
# Refresh file database
initexmf --update-fndb
initexmf --mklinks --force
```

### Alternative: Use Overleaf (NO INSTALLATION)
1. Go to https://www.overleaf.com/
2. Create free account
3. Upload `paper.tex` and `references.bib`
4. Click "Recompile" - works immediately!

## Quick Test

After installing packages, test with:
```powershell
pdflatex paper.tex
```

If you see lots of text output without errors, it's working!
