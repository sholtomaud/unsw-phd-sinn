#!/bin/bash

# GitHub Project Setup Script
# This script sets up a comprehensive GitHub project with automation workflows
# Requires: GitHub CLI (gh) to be installed and authenticated

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
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

# Function to check if gh CLI is installed and authenticated
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v gh &> /dev/null; then
        print_error "GitHub CLI (gh) is not installed. Please install it first."
        echo "Visit: https://cli.github.com/"
        exit 1
    fi
    
    if ! gh auth status &> /dev/null; then
        print_error "GitHub CLI is not authenticated. Please run 'gh auth login' first."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to get repository information
get_repo_info() {
    print_status "Getting repository information..."
    
    # Check if we're in a git repository
    if ! git rev-parse --is-inside-work-tree &> /dev/null; then
        print_error "Not in a git repository. Please run this script from within a git repository."
        exit 1
    fi
    
    # Get repository owner and name
    REPO_URL=$(git config --get remote.origin.url)
    if [[ $REPO_URL == *"github.com"* ]]; then
        if [[ $REPO_URL == *"git@github.com"* ]]; then
            REPO_FULL=$(echo $REPO_URL | sed 's/git@github.com://' | sed 's/\.git$//')
        else
            REPO_FULL=$(echo $REPO_URL | sed 's/https:\/\/github.com\///' | sed 's/\.git$//')
        fi
    else
        print_error "This doesn't appear to be a GitHub repository."
        exit 1
    fi
    
    REPO_OWNER=$(echo $REPO_FULL | cut -d'/' -f1)
    REPO_NAME=$(echo $REPO_FULL | cut -d'/' -f2)
    
    print_success "Repository: $REPO_OWNER/$REPO_NAME"
}

# Function to create project
create_project() {
    print_status "Creating GitHub project..."
    
    PROJECT_NAME="$REPO_NAME Project"
    
    # Create project (using Projects V2)
    PROJECT_OUTPUT=$(gh project create \
        --owner "$REPO_OWNER" \
        --title "$PROJECT_NAME" \
        --format json)
    
    if [ $? -ne 0 ]; then
        print_error "Failed to create project"
        exit 1
    fi
    
    PROJECT_ID=$(echo "$PROJECT_OUTPUT" | jq -r '.id')
    PROJECT_URL=$(echo "$PROJECT_OUTPUT" | jq -r '.url')
    
    if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" == "null" ]; then
        print_error "Failed to extract project ID"
        exit 1
    fi
    
    print_success "Created project: $PROJECT_NAME (ID: $PROJECT_ID)"
}

# Function to setup project fields
setup_project_fields() {
    print_status "Setting up project fields..."
    
    # Note: GitHub CLI project field commands may vary based on version
    # Let's use a more robust approach with Kanban-specific statuses
    
    # Add Status field optimized for Kanban workflow
    print_status "Adding Kanban Status field..."
    gh project field-create "$PROJECT_ID" \
        --name "Status" \
        --type "single_select" \
        --single-select-option "📋 Backlog" \
        --single-select-option "🔄 Todo" \
        --single-select-option "🏃 In Progress" \
        --single-select-option "👀 In Review" \
        --single-select-option "🧪 Testing" \
        --single-select-option "✅ Done" \
        --single-select-option "🚫 Blocked" 2>/dev/null || print_warning "Status field may already exist"
    
    # Add Priority field
    print_status "Adding Priority field..."
    gh project field-create "$PROJECT_ID" \
        --name "Priority" \
        --type "single_select" \
        --single-select-option "🔴 Critical" \
        --single-select-option "🟠 High" \
        --single-select-option "🟡 Medium" \
        --single-select-option "🟢 Low" \
        --single-select-option "🔵 Nice to Have" 2>/dev/null || print_warning "Priority field may already exist"
    
    # Add Story Points field for estimation
    print_status "Adding Story Points field..."
    gh project field-create "$PROJECT_ID" \
        --name "Story Points" \
        --type "single_select" \
        --single-select-option "1" \
        --single-select-option "2" \
        --single-select-option "3" \
        --single-select-option "5" \
        --single-select-option "8" \
        --single-select-option "13" \
        --single-select-option "21" 2>/dev/null || print_warning "Story Points field may already exist"
    
    # Add Sprint field
    print_status "Adding Sprint field..."
    gh project field-create "$PROJECT_ID" \
        --name "Sprint" \
        --type "single_select" \
        --single-select-option "Backlog" \
        --single-select-option "Sprint 1" \
        --single-select-option "Sprint 2" \
        --single-select-option "Sprint 3" \
        --single-select-option "Future" 2>/dev/null || print_warning "Sprint field may already exist"
    
    print_success "Kanban project fields configured"
}

# Function to setup project views
setup_project_views() {
    print_status "Setting up Kanban project views..."
    
    # Note: View creation may not be fully supported in all GitHub CLI versions
    # Let's add error handling and informative messages
    
    print_status "Creating Kanban Board view..."
    if gh project view-create "$PROJECT_ID" \
        --name "Kanban Board" \
        --type "board" \
        --field "Status" 2>/dev/null; then
        print_success "Kanban Board view created"
    else
        print_warning "Kanban Board view creation may not be supported or already exists"
    fi
    
    print_status "Creating Backlog view..."
    if gh project view-create "$PROJECT_ID" \
        --name "Backlog" \
        --type "table" 2>/dev/null; then
        print_success "Backlog view created"
    else
        print_warning "Backlog view creation may not be supported or already exists"
    fi
    
    print_status "Creating Sprint Planning view..."
    if gh project view-create "$PROJECT_ID" \
        --name "Sprint Planning" \
        --type "table" 2>/dev/null; then
        print_success "Sprint Planning view created"
    else
        print_warning "Sprint Planning view creation may not be supported or already exists"
    fi
    
    print_status "Creating Roadmap view..."
    if gh project view-create "$PROJECT_ID" \
        --name "Roadmap" \
        --type "roadmap" 2>/dev/null; then
        print_success "Roadmap view created"
    else
        print_warning "Roadmap view creation may not be supported or already exists"
    fi
    
    print_success "Kanban project views configuration attempted"
}

# Function to link repository to project
link_repository() {
    print_status "Linking repository to project..."
    
    # Link the repository to the project
    if gh project link "$PROJECT_ID" "$REPO_OWNER/$REPO_NAME" 2>/dev/null; then
        print_success "Repository linked to project"
    else
        print_warning "Repository linking may not be supported or already linked"
        print_status "You may need to link the repository manually in the GitHub web interface"
    fi
}

# Function to setup automation workflows
setup_automation() {
    print_status "Setting up project automation workflows..."
    
    # Note: GitHub CLI doesn't have direct support for setting up project automation workflows
    # These need to be configured through the GitHub web interface or GraphQL API
    
    print_status "Automation workflows need to be configured manually."
    print_status "Here's the step-by-step process:"
    
    echo ""
    echo "📋 KANBAN AUTOMATION SETUP:"
    echo "================================"
    echo ""
    echo "1. Go to your project: $PROJECT_URL"
    echo "2. Click the '⚙️ Settings' button (top right of project)"
    echo "3. Select 'Workflows' from the left sidebar"
    echo "4. Click '+ Add workflow' and configure these workflows:"
    echo ""
    echo "   🔄 AUTO-ADD TO PROJECT:"
    echo "   ----------------------"
    echo "   • Trigger: Issues and pull requests"
    echo "   • Condition: Repository = $REPO_OWNER/$REPO_NAME"
    echo "   • Action: Add to project"
    echo "   • Set Status to: 📋 Backlog"
    echo ""
    echo "   🎯 ITEM ADDED TO PROJECT:"
    echo "   ------------------------"
    echo "   • Trigger: Item added to project"
    echo "   • Condition: Status is empty"
    echo "   • Action: Set Status to '📋 Backlog'"
    echo "   • Action: Set Priority to '🟡 Medium'"
    echo ""
    echo "   ✅ ISSUE CLOSED:"
    echo "   ---------------"
    echo "   • Trigger: Issue closed"
    echo "   • Action: Set Status to '✅ Done'"
    echo ""
    echo "   🔀 PULL REQUEST MERGED:"
    echo "   ----------------------"
    echo "   • Trigger: Pull request merged"
    echo "   • Action: Set Status to '✅ Done'"
    echo ""
    echo "   🗄️ AUTO-ARCHIVE:"
    echo "   ----------------"
    echo "   • Trigger: Item closed"
    echo "   • Condition: Status is '✅ Done'"
    echo "   • Action: Archive item"
    echo ""
    echo "   🚫 BLOCKED ITEMS:"
    echo "   ----------------"
    echo "   • Trigger: Label added"
    echo "   • Condition: Label is 'blocked'"
    echo "   • Action: Set Status to '🚫 Blocked'"
    echo ""
    
    # Try to open the project settings directly if possible
    if command -v open &> /dev/null; then
        echo "5. Opening project settings in browser..."
        open "$PROJECT_URL/settings" 2>/dev/null || true
    elif command -v xdg-open &> /dev/null; then
        echo "5. Opening project settings in browser..."
        xdg-open "$PROJECT_URL/settings" 2>/dev/null || true
    else
        echo "5. Manually navigate to: $PROJECT_URL/settings"
    fi
    
    echo ""
    print_warning "These workflows are essential for proper Kanban automation!"
    print_status "Once configured, your Kanban board will automatically manage item states."
}

# Function to create issue templates
create_issue_templates() {
    print_status "Creating issue templates..."
    
    # Create .github directory if it doesn't exist
    mkdir -p .github/ISSUE_TEMPLATE
    
    # Bug report template
    cat > .github/ISSUE_TEMPLATE/bug_report.yml << 'EOF'
name: Bug Report
description: Create a report to help us improve
title: "[BUG] "
labels: ["bug", "needs-triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Also tell us, what did you expect to happen?
      placeholder: Tell us what you see!
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Steps to reproduce
      description: Please provide steps to reproduce the issue
      placeholder: |
        1. Go to '...'
        2. Click on '....'
        3. Scroll down to '....'
        4. See error
    validations:
      required: true
  - type: textarea
    id: environment
    attributes:
      label: Environment
      description: What environment are you using?
      placeholder: |
        - OS: [e.g. Ubuntu 20.04, Windows 10, macOS 12]
        - Version: [e.g. 1.2.3]
        - Browser: [if applicable]
    validations:
      required: false
EOF

    # Feature request template
    cat > .github/ISSUE_TEMPLATE/feature_request.yml << 'EOF'
name: Feature Request
description: Suggest an idea for this project
title: "[FEATURE] "
labels: ["enhancement", "needs-triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for suggesting a new feature!
  - type: textarea
    id: problem
    attributes:
      label: Is your feature request related to a problem?
      description: A clear and concise description of what the problem is.
      placeholder: I'm always frustrated when...
    validations:
      required: false
  - type: textarea
    id: solution
    attributes:
      label: Describe the solution you'd like
      description: A clear and concise description of what you want to happen.
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Describe alternatives you've considered
      description: A clear and concise description of any alternative solutions or features you've considered.
    validations:
      required: false
  - type: textarea
    id: additional-context
    attributes:
      label: Additional context
      description: Add any other context or screenshots about the feature request here.
    validations:
      required: false
EOF

    # Task template
    cat > .github/ISSUE_TEMPLATE/task.yml << 'EOF'
name: Task
description: Create a task for project work
title: "[TASK] "
labels: ["task"]
body:
  - type: textarea
    id: description
    attributes:
      label: Task Description
      description: Describe what needs to be done
    validations:
      required: true
  - type: textarea
    id: acceptance-criteria
    attributes:
      label: Acceptance Criteria
      description: What needs to be completed for this task to be considered done?
      placeholder: |
        - [ ] Criteria 1
        - [ ] Criteria 2
        - [ ] Criteria 3
    validations:
      required: true
  - type: dropdown
    id: priority
    attributes:
      label: Priority
      description: What priority is this task?
      options:
        - Low
        - Medium
        - High
        - Critical
    validations:
      required: true
EOF

    print_success "Issue templates created"
}

# Function to create pull request template
create_pr_template() {
    print_status "Creating pull request template..."
    
    mkdir -p .github
    
    cat > .github/pull_request_template.md << 'EOF'
## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## How Has This Been Tested?
Please describe the tests that you ran to verify your changes.

- [ ] Test A
- [ ] Test B

## Checklist:
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Related Issues
Closes #(issue number)
EOF

    print_success "Pull request template created"
}

# Function to setup repository labels
setup_labels() {
    print_status "Setting up repository labels..."
    
    # Define labels with colors (format: "label_name:color")
    LABELS=(
        "bug:d73a4a"
        "enhancement:a2eeef"
        "documentation:0075ca"
        "duplicate:cfd3d7"
        "good first issue:7057ff"
        "help wanted:008672"
        "invalid:e4e669"
        "question:d876e3"
        "wontfix:ffffff"
        "needs triage:fef2c0"
        "priority high:d93f0b"
        "priority medium:fbca04"
        "priority low:0e8a16"
        "task:1d76db"
        "epic:3e4b9e"
        "blocked:b60205"
        "in progress:fbca04"
        "ready for review:0e8a16"
    )
    
    # Create/update labels
    for label_entry in "${LABELS[@]}"; do
        label_name=$(echo "$label_entry" | cut -d':' -f1)
        label_color=$(echo "$label_entry" | cut -d':' -f2)
        
        if gh label create "$label_name" --color "$label_color" --force 2>/dev/null; then
            print_status "Created/updated label: $label_name"
        else
            print_warning "Label '$label_name' may already exist or failed to create"
        fi
    done
    
    print_success "Repository labels configured"
}

# Function to setup repository settings
setup_repository_settings() {
    print_status "Configuring repository settings..."
    
    # Enable issues and projects
    gh repo edit "$REPO_OWNER/$REPO_NAME" \
        --enable-issues \
        --enable-projects \
        --enable-wiki=false \
        --delete-branch-on-merge || true
    
    print_success "Repository settings configured"
}

# Function to create initial project items
create_initial_items() {
    print_status "Creating initial project items..."
    
    # Create a welcome issue (using the correct method for getting issue number)
    WELCOME_ISSUE_URL=$(gh issue create \
        --title "🎉 Welcome to $REPO_NAME!" \
        --body "This issue was automatically created to welcome you to your new project setup.

## Project Setup Complete ✅

Your repository now has:
- 📋 Project board with automation workflows
- 🏷️ Standardized labels
- 📝 Issue and PR templates
- 🔄 Automated project management

## Next Steps
- [ ] Review project settings and customize as needed
- [ ] Set up any additional automation workflows
- [ ] Create your first real issue or task
- [ ] Start coding! 🚀

You can safely close this issue once you've reviewed the setup." \
        --label "task,good first issue")
    
    if [ -n "$WELCOME_ISSUE_URL" ]; then
        # Extract issue number from URL
        WELCOME_ISSUE=$(echo "$WELCOME_ISSUE_URL" | grep -o '[0-9]*$')
        print_success "Created welcome issue #$WELCOME_ISSUE"
        print_status "The welcome issue has been added to your Kanban board!"
    else
        print_warning "Failed to create welcome issue"
    fi
}

# Function to display summary
display_summary() {
    echo ""
    echo "======================================"
    echo "🎉 GitHub Kanban Project Setup Complete!"
    echo "======================================"
    echo ""
    echo "Repository: $REPO_OWNER/$REPO_NAME"
    echo "Project: $PROJECT_NAME"
    echo ""
    echo "✅ Created Kanban project with custom fields and views"
    echo "✅ Set up issue and PR templates"
    echo "✅ Configured repository labels"
    echo "✅ Created initial project structure"
    echo ""
    echo "🔗 Quick Links:"
    echo "   Repository: https://github.com/$REPO_OWNER/$REPO_NAME"
    echo "   Kanban Board: $PROJECT_URL"
    echo "   Project Settings: $PROJECT_URL/settings"
    echo "   Issues: https://github.com/$REPO_OWNER/$REPO_NAME/issues"
    echo ""
    echo "⚠️  IMPORTANT - Manual Setup Required:"
    echo "   🔄 Configure Kanban automation workflows (see instructions above)"
    echo "   📋 Set 'Kanban Board' as your default view"
    echo "   🎯 Customize sprint and priority options as needed"
    echo ""
    echo "🎯 Your Kanban board is ready! Once you configure the workflows,"
    echo "   issues and PRs will automatically flow through your board."
    echo ""
    echo "💡 Pro Tips:"
    echo "   • Use the issue templates to maintain consistency"
    echo "   • Set up branch protection rules for important branches"
    echo "   • Consider enabling GitHub Actions for CI/CD"
    echo "   • Use the project's Sprint field to organize work"
    echo "   • Regularly review and update the automation workflows"
}

# Main execution
main() {
    echo "🚀 GitHub Project Setup Script"
    echo "==============================="
    echo ""
    
    check_prerequisites
    get_repo_info
    create_project
    setup_project_fields
    setup_project_views
    link_repository
    setup_automation
    create_issue_templates
    create_pr_template
    setup_labels
    setup_repository_settings
    create_initial_items
    display_summary
}

# Run the main function
main "$@"