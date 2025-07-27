#!/bin/bash

# MLflow Setup Script for Iris Classification Pipeline
# This script sets up MLflow tracking server and UI for hyperparameter tuning experiments
# Author: Abhyudaya B Tharakan 22f3001492

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MLFLOW_PORT=${MLFLOW_PORT:-5000}
MLFLOW_HOST=${MLFLOW_HOST:-0.0.0.0}
MLFLOW_BACKEND_STORE_URI=${MLFLOW_BACKEND_STORE_URI:-"sqlite:///mlflow.db"}
MLFLOW_ARTIFACT_ROOT=${MLFLOW_ARTIFACT_ROOT:-"./mlruns"}
MLFLOW_WORKERS=${MLFLOW_WORKERS:-1}

# Functions
print_header() {
    echo -e "${BLUE}=================================================${NC}"
    echo -e "${BLUE} MLflow Setup Script for Week-6 Assignment${NC}"
    echo -e "${BLUE}=================================================${NC}"
    echo ""
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    # Check if pip is available
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is not installed or not in PATH"
        exit 1
    fi
    
    print_success "Dependencies check completed"
}

check_mlflow_installation() {
    print_info "Checking MLflow installation..."
    
    if python3 -c "import mlflow" &> /dev/null; then
        MLFLOW_VERSION=$(python3 -c "import mlflow; print(mlflow.__version__)")
        print_success "MLflow is installed (version: $MLFLOW_VERSION)"
    else
        print_warning "MLflow is not installed. Installing from requirements.txt..."
        pip3 install mlflow optuna
        if python3 -c "import mlflow" &> /dev/null; then
            print_success "MLflow installation completed"
        else
            print_error "Failed to install MLflow"
            exit 1
        fi
    fi
}

setup_mlflow_directories() {
    print_info "Setting up MLflow directories..."
    
    # Create mlruns directory if it doesn't exist
    if [ ! -d "$MLFLOW_ARTIFACT_ROOT" ]; then
        mkdir -p "$MLFLOW_ARTIFACT_ROOT"
        print_success "Created MLflow artifacts directory: $MLFLOW_ARTIFACT_ROOT"
    else
        print_info "MLflow artifacts directory already exists: $MLFLOW_ARTIFACT_ROOT"
    fi
    
    # Create logs directory
    if [ ! -d "logs" ]; then
        mkdir -p "logs"
        print_success "Created logs directory"
    fi
}

create_mlflow_startup_script() {
    print_info "Creating MLflow startup script..."
    
    cat > start_mlflow.sh << 'EOF'
#!/bin/bash

# MLflow Server Startup Script
# This script starts the MLflow tracking server in the background

MLFLOW_PORT=${MLFLOW_PORT:-5000}
MLFLOW_HOST=${MLFLOW_HOST:-0.0.0.0}
MLFLOW_BACKEND_STORE_URI=${MLFLOW_BACKEND_STORE_URI:-"sqlite:///mlflow.db"}
MLFLOW_ARTIFACT_ROOT=${MLFLOW_ARTIFACT_ROOT:-"./mlruns"}
MLFLOW_WORKERS=${MLFLOW_WORKERS:-1}

echo "Starting MLflow server..."
echo "Host: $MLFLOW_HOST"
echo "Port: $MLFLOW_PORT"
echo "Backend Store URI: $MLFLOW_BACKEND_STORE_URI"
echo "Artifact Root: $MLFLOW_ARTIFACT_ROOT"
echo "Workers: $MLFLOW_WORKERS"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start MLflow server in background
mlflow server \
    --host $MLFLOW_HOST \
    --port $MLFLOW_PORT \
    --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
    --default-artifact-root $MLFLOW_ARTIFACT_ROOT \
    --workers $MLFLOW_WORKERS \
    > logs/mlflow_server.log 2>&1 &

MLFLOW_PID=$!
echo "MLflow server started with PID: $MLFLOW_PID"
echo $MLFLOW_PID > logs/mlflow_server.pid

# Wait a moment for server to start
sleep 3

# Check if server is running
if ps -p $MLFLOW_PID > /dev/null 2>&1; then
    echo "MLflow server is running successfully!"
    echo "You can access the MLflow UI at: http://localhost:$MLFLOW_PORT"
    echo "To stop the server, run: ./stop_mlflow.sh"
else
    echo "Failed to start MLflow server. Check logs/mlflow_server.log for details."
    exit 1
fi
EOF
    
    chmod +x start_mlflow.sh
    print_success "MLflow startup script created: start_mlflow.sh"
}

create_mlflow_stop_script() {
    print_info "Creating MLflow stop script..."
    
    cat > stop_mlflow.sh << 'EOF'
#!/bin/bash

# MLflow Server Stop Script
# This script stops the MLflow tracking server

if [ -f "logs/mlflow_server.pid" ]; then
    PID=$(cat logs/mlflow_server.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Stopping MLflow server (PID: $PID)..."
        kill $PID
        
        # Wait for process to stop
        sleep 2
        
        if ! ps -p $PID > /dev/null 2>&1; then
            echo "MLflow server stopped successfully"
            rm -f logs/mlflow_server.pid
        else
            echo "Force killing MLflow server..."
            kill -9 $PID
            rm -f logs/mlflow_server.pid
        fi
    else
        echo "MLflow server is not running (PID $PID not found)"
        rm -f logs/mlflow_server.pid
    fi
else
    echo "MLflow server PID file not found. Checking for any running MLflow processes..."
    
    # Find and kill any MLflow processes
    MLFLOW_PIDS=$(pgrep -f "mlflow server" || true)
    if [ -n "$MLFLOW_PIDS" ]; then
        echo "Found MLflow processes: $MLFLOW_PIDS"
        echo "Killing MLflow processes..."
        kill $MLFLOW_PIDS
        sleep 2
        
        # Force kill if still running
        MLFLOW_PIDS=$(pgrep -f "mlflow server" || true)
        if [ -n "$MLFLOW_PIDS" ]; then
            echo "Force killing remaining MLflow processes..."
            kill -9 $MLFLOW_PIDS
        fi
        
        echo "MLflow server stopped"
    else
        echo "No MLflow server processes found"
    fi
fi
EOF
    
    chmod +x stop_mlflow.sh
    print_success "MLflow stop script created: stop_mlflow.sh"
}

show_usage_instructions() {
    echo ""
    print_header
    echo -e "${GREEN}MLflow Setup Completed Successfully!${NC}"
    echo ""
    echo -e "${BLUE}Quick Start:${NC}"
    echo "1. Start MLflow server:"
    echo "   ./start_mlflow.sh"
    echo ""
    echo "2. Run the demo script:"
    echo "   python demo_mlflow.py"
    echo ""
    echo "3. Access MLflow UI:"
    echo "   http://localhost:$MLFLOW_PORT"
    echo ""
    echo -e "${BLUE}MLflow Management:${NC}"
    echo "• Start server: ./start_mlflow.sh"
    echo "• Stop server:  ./stop_mlflow.sh"
    echo "• Server logs:  logs/mlflow_server.log"
    echo ""
}

main() {
    print_header
    
    # Run setup steps
    check_dependencies
    check_mlflow_installation
    setup_mlflow_directories
    create_mlflow_startup_script
    create_mlflow_stop_script
    
    show_usage_instructions
}

# Run main function
main "$@"