#!/bin/bash

# This script deletes all projects (boards) from your GitHub account using the GitHub CLI.
# Please exercise extreme caution as this action is irreversible.

# Function to display an error and exit
function error_exit {
    echo "Error: $1" >&2
    exit 1
}

echo "---"
echo "WARNING: This script will delete ALL projects from your GitHub account."
echo "This action is irreversible. Please proceed with extreme caution."
echo "---"

read -p "Are you absolutely sure you want to delete all projects? (yes/no): " confirmation
if [[ "$confirmation" != "yes" ]]; then
    echo "Operation aborted. No projects were deleted."
    exit 0
fi

echo "---"
echo "Fetching all projects in your account..."
echo "---"

# Get a list of all project numbers
# We're specifically targeting user-owned projects, not organization projects.
project_json=$(gh project list --owner "@me" --format json)

# Check if the JSON output is empty or not a valid array structure with projects
if [[ -z "$project_json" || ! $(echo "$project_json" | jq -e '.projects | arrays') ]]; then
    echo "No projects found in your account or unexpected JSON structure."
    # For debugging, you could uncomment the next line:
    # echo "Full JSON output received was: $project_json"
    exit 0
fi

# CORRECTED JQ COMMAND: Access the 'projects' array and extract the 'number'
project_numbers=$(echo "$project_json" | jq -r '.projects[].number')

# Check if any project numbers were actually extracted
if [ -z "$project_numbers" ]; then
    echo "No project numbers could be extracted from the output. This might indicate that the 'projects' array was empty or individual projects lacked a 'number' field."
    echo "Full JSON output received was:"
    echo "$project_json"
    exit 0
fi

echo "Found the following project numbers to delete:"
echo "$project_numbers"
echo "---"

read -p "Confirm one more time: Are you absolutely sure you want to delete these projects? Type 'DELETE' to confirm: " final_confirmation
if [[ "$final_confirmation" != "DELETE" ]]; then
    echo "Operation aborted. No projects were deleted."
    exit 0
fi

echo "---"
echo "Deleting projects..."
echo "---"

# Loop through each project number and delete it
for num in $project_numbers; do
    echo "Attempting to delete project with number: $num"
    # Use --owner @me explicitly when deleting by number
    if gh project delete "$num" --owner "@me"; then
        echo "Successfully deleted project number: $num"
    else
        error_exit "Failed to delete project number: $num. Please check your GitHub CLI authentication and permissions."
    fi
done

echo "---"
echo "All accessible projects have been processed."
echo "---"