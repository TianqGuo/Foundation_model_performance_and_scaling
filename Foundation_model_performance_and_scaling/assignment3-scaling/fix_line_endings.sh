#!/bin/bash
# ==============================================================================
# Fix Line Endings for All Shell Scripts
# ==============================================================================
#
# Converts all .sh files from Windows (CRLF) to Unix (LF) line endings
# and makes them executable.
#
# USAGE:
#   ./fix_line_endings.sh
#
# ==============================================================================

echo "Fixing line endings for all shell scripts..."

# Find all .sh files and fix line endings
find . -type f -name "*.sh" | while read -r file; do
    echo "  Processing: $file"
    sed -i 's/\r$//' "$file"
    chmod +x "$file"
done

echo "Done! All shell scripts fixed."