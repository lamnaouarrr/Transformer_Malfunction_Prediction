#!/bin/bash

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo "=================================================="
echo "      Python Environment Setup Script"
echo "=================================================="

# Check if a requirements file was provided
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}No requirements file specified. Using default 'requirements.txt'${NC}"
    REQUIREMENTS_FILE="requirements.txt"
else
    REQUIREMENTS_FILE="$1"
fi

echo "Requirements file: $REQUIREMENTS_FILE"

# Check if the requirements file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${RED}Error: Requirements file '$REQUIREMENTS_FILE' not found!${NC}"
    echo "Please make sure the file exists in the current directory."
    exit 1
fi

# Check if Python is installed
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
    echo -e "${GREEN}✓ Python 3 found${NC}"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
    echo -e "${GREEN}✓ Python found${NC}"
else
    echo -e "${RED}❌ Python not found!${NC}"
    echo -e "${YELLOW}Installing Python...${NC}"
    
    # Detect the operating system
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
    elif type lsb_release >/dev/null 2>&1; then
        OS=$(lsb_release -si)
    elif [ -f /etc/lsb-release ]; then
        . /etc/lsb-release
        OS=$DISTRIB_ID
    else
        OS=$(uname -s)
    fi
    
    # Install Python based on the operating system
    case "$OS" in
        *Ubuntu*|*Debian*)
            echo "Detected Ubuntu/Debian-based system"
            sudo apt-get update
            sudo apt-get install -y python3 python3-pip
            PYTHON_CMD="python3"
            ;;
        *CentOS*|*RHEL*|*Fedora*)
            echo "Detected CentOS/RHEL/Fedora-based system"
            sudo yum install -y python3 python3-pip
            PYTHON_CMD="python3"
            ;;
        *Alpine*)
            echo "Detected Alpine Linux"
            apk add --update python3 py3-pip
            PYTHON_CMD="python3"
            ;;
        *)
            echo -e "${RED}Unsupported operating system: $OS${NC}"
            echo "Please install Python manually and run this script again."
            echo "Installation instructions: https://www.python.org/downloads/"
            exit 1
            ;;
    esac
    
    # Verify Python installation
    if command -v $PYTHON_CMD &>/dev/null; then
        echo -e "${GREEN}✓ Python successfully installed${NC}"
    else
        echo -e "${RED}❌ Failed to install Python automatically${NC}"
        echo "Please install Python manually and run this script again."
        echo "Installation instructions: https://www.python.org/downloads/"
        exit 1
    fi
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR_VERSION=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR_VERSION=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

echo "Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR_VERSION" -lt 3 ] || ( [ "$PYTHON_MAJOR_VERSION" -eq 3 ] && [ "$PYTHON_MINOR_VERSION" -lt 7 ] ); then
    echo -e "${YELLOW}⚠ Warning: Python $PYTHON_VERSION detected. Python 3.7+ is recommended.${NC}"
else
    echo -e "${GREEN}✓ Python version is sufficient${NC}"
fi

# Check if pip is installed
if $PYTHON_CMD -m pip --version &>/dev/null; then
    echo -e "${GREEN}✓ pip is installed${NC}"
else
    echo -e "${YELLOW}Installing pip...${NC}"
    
    # Try to install pip
    $PYTHON_CMD -m ensurepip --upgrade || {
        # If ensurepip fails, download get-pip.py
        echo "Downloading get-pip.py..."
        curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        $PYTHON_CMD get-pip.py
        rm -f get-pip.py
    }
    
    # Verify pip installation
    if $PYTHON_CMD -m pip --version &>/dev/null; then
        echo -e "${GREEN}✓ pip successfully installed${NC}"
    else
        echo -e "${RED}❌ Failed to install pip${NC}"
        echo "Please install pip manually and run this script again."
        exit 1
    fi
fi

# Upgrade pip
echo "Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip

# Install requirements
echo -e "\nInstalling packages from $REQUIREMENTS_FILE..."
if $PYTHON_CMD -m pip install -r "$REQUIREMENTS_FILE"; then
    echo -e "${GREEN}✓ Successfully installed all required packages!${NC}"
else
    echo -e "${RED}❌ Failed to install some packages${NC}"
    echo "Try installing problematic packages individually."
fi

# Check if TensorFlow was installed and has GPU support
if $PYTHON_CMD -c "import tensorflow" &>/dev/null; then
    echo -e "\nChecking for GPU support in TensorFlow..."
    GPU_INFO=$($PYTHON_CMD -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" 2>/dev/null)
    
    if [[ $GPU_INFO == *"PhysicalDevice"* ]]; then
        echo -e "${GREEN}✓ TensorFlow GPU support is available!${NC}"
    else
        echo -e "${YELLOW}⚠ TensorFlow is installed but GPU support is not available${NC}"
        echo "For deep learning with large datasets, GPU acceleration is recommended."
    fi
fi

echo -e "\n${GREEN}Setup completed!${NC}"
echo "You can now run your project."