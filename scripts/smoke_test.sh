#!/bin/bash

# Horizon AI Assistant - Smoke Test Script
# Simple health check for the local server

set -e

echo "🚀 Horizon AI Assistant - Smoke Test"
echo "===================================="

# Configuration
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
URL="http://${HOST}:${PORT}"
TIMEOUT=10

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "ok" ]; then
        echo -e "${GREEN}✅ ${message}${NC}"
    elif [ "$status" = "warn" ]; then
        echo -e "${YELLOW}⚠️  ${message}${NC}"
    else
        echo -e "${RED}❌ ${message}${NC}"
    fi
}

echo "🔍 Testing server at: $URL"
echo "⏱️  Timeout: ${TIMEOUT}s"
echo ""

# Test 1: Basic connectivity
echo "1. Testing basic connectivity..."
if curl -s --connect-timeout $TIMEOUT "$URL" > /dev/null 2>&1; then
    print_status "ok" "Server is reachable"
else
    print_status "error" "Server is not reachable at $URL"
    echo ""
    echo "💡 Tips:"
    echo "   • Make sure the server is running: python3 app.py"
    echo "   • Check if port $PORT is correct"
    echo "   • Verify no firewall is blocking the connection"
    exit 1
fi

# Test 2: HTTP response code
echo ""
echo "2. Testing HTTP response..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT "$URL")
if [ "$HTTP_CODE" = "200" ]; then
    print_status "ok" "Server returns HTTP 200 OK"
elif [ "$HTTP_CODE" = "000" ]; then
    print_status "error" "Connection failed (HTTP 000)"
    exit 1
else
    print_status "warn" "Server returns HTTP $HTTP_CODE (expected 200)"
fi

# Test 3: Content check
echo ""
echo "3. Testing page content..."
CONTENT=$(curl -s --connect-timeout $TIMEOUT "$URL" 2>/dev/null || echo "")
if echo "$CONTENT" | grep -q "Horizon AI Assistant" > /dev/null 2>&1; then
    print_status "ok" "Page contains expected title"
else
    print_status "warn" "Page content may be unexpected"
fi

# Test 4: JavaScript/CSS resources
echo ""
echo "4. Testing static resources..."
if echo "$CONTENT" | grep -q "ThemeManager" > /dev/null 2>&1; then
    print_status "ok" "Theme system appears to be loaded"
else
    print_status "warn" "Theme system may not be loaded"
fi

# Test 5: API endpoints (optional)
echo ""
echo "5. Testing API endpoints..."
API_ENDPOINTS=("/api/calendar/health" "/api/notes/list")

for endpoint in "${API_ENDPOINTS[@]}"; do
    api_url="${URL}${endpoint}"
    api_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "$api_url" 2>/dev/null || echo "000")
    if [ "$api_code" = "200" ]; then
        print_status "ok" "API endpoint $endpoint is working"
    elif [ "$api_code" = "404" ]; then
        print_status "warn" "API endpoint $endpoint not found (may be optional)"
    else
        print_status "warn" "API endpoint $endpoint returns HTTP $api_code"
    fi
done

echo ""
echo "🏁 Smoke test completed!"
echo ""
echo "📊 Summary:"
echo "   • Server URL: $URL"
echo "   • HTTP Status: $HTTP_CODE"
echo "   • Connectivity: OK"
echo ""
echo "🎯 Next steps:"
echo "   • Open $URL in your browser"
echo "   • Test theme switching with Ctrl+Shift+T"
echo "   • Try voice commands or chat functionality"
echo ""
echo "💡 For issues, check server logs or README.md"