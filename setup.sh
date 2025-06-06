#!/bin/bash
set -e
set -u

log() {
    local GREEN="\033[1;32m"
    local BLUE="\033[1;34m"
    local CYAN="\033[1;36m"
    local WHITE="\033[1;37m"
    local RESET="\033[0m"

    local timestamp="${GREEN}$(date '+%Y-%m-%d %H:%M:%S.%3N')${RESET}"
    local level="${BLUE}DEBUG${RESET}"
    local caller="${CYAN}${BASH_SOURCE[1]##*/}:${BASH_LINENO[0]}${RESET}"
    local message="${WHITE}$*${RESET}"

    echo -e "${timestamp} | ${level} | ${caller} - ${message}"
}

create_safe() {
    local ENV_NAME=""
    local args=("$@")

    # Extract the environment name from -n or --name
    for ((i = 0; i < ${#args[@]}; i++)); do
        if [[ "${args[i]}" == "-n" || "${args[i]}" == "--name" ]]; then
            ENV_NAME="${args[i + 1]}"
            break
        fi
    done

    # Remove if it exists
    if conda info --envs | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
        conda remove -n "$ENV_NAME" --all -y
    fi

    log "Creating environment '$ENV_NAME'..."
    conda create "${args[@]}"

    log "✅ Environment '$ENV_NAME' created successfully."
}

WORKDIR=$(pwd)
LIB_DIR="$WORKDIR/lib"
mkdir -p "$LIB_DIR"

# STEP 1
# Create semCRC environment
log "Creating semCRC environment..."
conda create -y -n semcrc python=3.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semcrc

# Install PyTorch and CUDA
log "Installing PyTorch and CUDA..."
python -m pip install \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

# STEP 2
# Download and install K-RCPS
log "Installing K-RCPS..."
cd "$WORKDIR"
KRCPS_DIR="$LIB_DIR/krcps"

if [ -d "$KRCPS_DIR/.git" ]; then
    log "K-RCPS already exists — pulling latest changes..."
    cd "$KRCPS_DIR"
    git fetch origin
    git reset --hard origin/main
else
    log "Downloading K-RCPS..."
    git clone https://github.com/Sulam-Group/k-rcps.git "$KRCPS_DIR"
fi

cd "$KRCPS_DIR"
python -m pip install -e .

# Install additional dependencies
log "installing additional dependencies..."
python -m pip install -r requirements.txt

# STEP 3
# Download and install SuPreM
# Create SuPreM environment
log "Creating SuPreM environment..."
create_safe -y -n suprem python=3.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate suprem

# Install PyTorch and CUDA for SuPreM
# taken from https://github.com/MrGiovanni/SuPreM/tree/main/direct_inference
python -m pip install \
    torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install monai[all]==0.9.0

log "Installing SuPreM..."
cd "$WORKDIR"
SUPREM_DIR="$LIB_DIR/suprem"

if [ -d "$SUPREM_DIR/.git" ]; then
    log "SuPreM already exists - pulling latest changes..."
    cd "$SUPREM_DIR"
    git fetch origin
    git reset --hard origin/main
else
    log "Downloading SuPreM..."
    git clone https://github.com/JacopoTeneggi/SuPreM.git "$SUPREM_DIR"
fi

cd "$SUPREM_DIR"
python -m pip install -r requirements.txt

# STEP 3
# Download and install ODL
log "Creating ODL environment..."
create_safe -y -n odl python=3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate odl

# Install ODL
log "Installing ODL..."
conda install -y conda-forge::odl matplotlib pytest scikit-image spyder

# Install PyTorch and CUDA
log "Installing PyTorch and CUDA..."
python -m pip install \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install Astra Toolbox
log "Installing Astra Toolbox..."
conda install -y -c astra-toolbox -c nvidia astra-toolbox
