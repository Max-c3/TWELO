# Identification and Obfuscation of personal data in drone images
This repository contains the code for the graduation project for the *Data and ML* Bootcamp by Le Wagon.

## Setup
To set up the repository on your machine, simply follow this:
```
git clone git@github.com:Max-c3/Kestrix_Project.git
make setup_virtual_env
make install
```

## Workflow for updating from main
1. switch to main branch
- `git switch main`
2. Pull changes from github
- `git pull origin main`
3. Reinstall package
- `make reinstall_package`
4. Either create new branch to work on or switch to the one you were working on
- If latter:
  - `git switch <branch-name>`
  - `git merge main`
