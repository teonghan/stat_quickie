#!/bin/bash

# --------------------------------------
# Streamlit App: Universal Installer for macOS (Intel + ARM64)
# Uses environment.yml for conda setup
# --------------------------------------

ENV_NAME="statquickie"
APP_DIR="$(pwd)"
SHORTCUT_NAME="Run Basic Stat Quickie App"
YAML_FILE="$APP_DIR/environment.yml"

echo "---------------------------------------------"
echo "🧠 Detecting architecture..."
echo "---------------------------------------------"

ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    echo "✅ Detected Apple Silicon (arm64)"
    MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
else
    echo "✅ Detected Intel (x86_64)"
    MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
fi

echo "---------------------------------------------"
echo "Checking Miniforge installation..."
echo "---------------------------------------------"

if [ ! -d "$HOME/miniforge3" ]; then
    echo "Downloading Miniforge for $ARCH..."
    curl -L -o Miniforge3-Installer.sh "$MINIFORGE_URL"
    chmod +x Miniforge3-Installer.sh
    echo "Installing Miniforge (no admin required)..."
    bash Miniforge3-Installer.sh -b -p $HOME/miniforge3
    rm Miniforge3-Installer.sh
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
else
    echo "Miniforge already installed."
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
fi

echo "---------------------------------------------"
echo "Setting up conda environment from environment.yml..."
echo "---------------------------------------------"

if [ ! -f "$YAML_FILE" ]; then
    echo "❌ ERROR: environment.yml not found in $APP_DIR"
    exit 1
fi

if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Creating environment $ENV_NAME from $YAML_FILE..."
    conda env create -f "$YAML_FILE"
else
    echo "Conda environment $ENV_NAME already exists."
fi

# ---------------------------------------------
echo "Creating launcher script: run_app.sh"
# ---------------------------------------------

RUNSH="$APP_DIR/run_app.sh"
cat > "$RUNSH" <<EOF
#!/bin/bash
source \$HOME/miniforge3/bin/activate $ENV_NAME
cd "$APP_DIR"
streamlit run app.py
read -p "Press ENTER to exit the app..."
EOF
chmod +x "$RUNSH"

# ---------------------------------------------
echo "Creating Automator app on Desktop..."
# ---------------------------------------------

AUTOMATOR_SCRIPT="$HOME/Desktop/$SHORTCUT_NAME.app"
APP_TMP="$HOME/Desktop/$SHORTCUT_NAME.scpt"

APPLESCRIPT=$(cat <<END
on run
    do shell script "cd '$APP_DIR'; '$RUNSH'"
end run
END
)

echo "$APPLESCRIPT" > "$APP_TMP"
osacompile -o "$AUTOMATOR_SCRIPT" "$APP_TMP"
rm "$APP_TMP"

# ---------------------------------------------
echo "Applying custom icon (if available)..."
# ---------------------------------------------

ICON_SOURCE="$APP_DIR/icon.icns"
ICON_DEST="$AUTOMATOR_SCRIPT/Contents/Resources/applet.icns"

if [ -f "$ICON_SOURCE" ]; then
    echo "Copying icon.icns to Automator app..."
    cp "$ICON_SOURCE" "$ICON_DEST"
    touch "$AUTOMATOR_SCRIPT"
else
    echo "icon.icns not found — skipping."
fi

# ---------------------------------------------
echo ""
echo "✅ Setup complete!"
echo "➡ A shortcut has been created on your Desktop:"
echo "   $SHORTCUT_NAME"
echo "📦 Double-click it to launch your Streamlit app."
echo "🛡 If prompted by macOS security, allow the app to run."
echo "---------------------------------------------"
