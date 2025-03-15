# Setting the Default Branch on GitHub

To make your AI-Powered FinTech Platform code visible on GitHub, follow these steps to set the `master` branch as the default branch:

1. Go to your GitHub repository: https://github.com/rafeeqqq/XNL-21BCE3918-LLM-1

2. Click on the "Settings" tab near the top of the repository page.

3. In the left sidebar, click on "Branches".

4. Under "Default branch", click on the dropdown menu that currently shows "main".

5. Select "master" from the dropdown menu.

6. Click on the "Update" button to save your changes.

7. Confirm the change in the dialog that appears.

After completing these steps, when you visit your repository, GitHub will display the contents of the `master` branch by default, which contains all your AI-Powered FinTech Platform code.

## Alternatively: Force Push to Main Branch

If you prefer to have your code on the `main` branch instead, you can run the following commands:

```bash
# Switch to the master branch
git checkout master

# Create a new main branch based on master
git branch -D main
git checkout -b main

# Force push the new main branch to GitHub
git push -f origin main

# Set main as the default branch on GitHub (follow steps 1-7 above but select "main" instead)
```

This will replace the current `main` branch with your `master` branch content.
