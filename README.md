# Project

This project builds a full ML pipeline to detect abnormal vessel trajectories from AIS data for two different routes.
It includes preprocessing, trajectory structuring, manual + rule-based labeling, experiments with 4 classical models and an LSTM, and visual tools for analysis.

## Participants
- [anokhver](https://github.com/anokhver) (Veronika Anokhina)
- [PjoterC](https://github.com/PjoterC) (Piotr Ciupiński)
- [Mironiusz](https://github.com/Mironiusz) (Rafał Mironko)

## Models
After its evaluation by the models, they obtained following scores (route 1 / route 2):
- Random forest: F1 = 0.992/0.972
- One-class SVM: F1 = 0.941/0.900
- Isolation forest: F1 = 0.884/0.875
- Logistic regression: F1 = 0.859/0.760
- LSTM: F1 = 0.879/0.807

## Features
- AIS data cleaning & feature engineering
- Automatic + manual abnormal-trajectory labeling
- Comparative evaluation across all 5 models
- LSTM sequence modeling for time-dependent patterns
- Metrics, plots, and experiment logs
- Lightweight GUI for trajectory visualization
- Dockerized environment

---
# Installing and running the application
In order to install the application, first clone the repository using git, and store it somewhere on your device. The application requires docker to run - since it was said in the requirements that docker-based implementation is preferred, we assume that docker is already installed.

## To install the application: 

### If you're using vscode with dev containers extention:
Simply open the **group05** directory with vscode, it should automatically detect the dockerfile and provide you with an option to build and run the container. If nothing shows, try CTRL+SHIFT+P and look for the "Open folder in container" command and click it.

### If you're not using vscode:
Run command line in the project directory (group05)
Use these 2 commands:
docker build -t django-vue-dev -f .devcontainer/Dockerfile .
docker run -it --rm -p 8000:8000 -p 5173:5173 -v ${PWD}:/workspace -w /workspace django-vue-dev

## After the container is running:
In the workspace directory of the container (default after starting), simply run the start script with command "./start.sh". It will run both the backend and the frontend of the application, making it ready to use.

If for some reason you don't have required permissions to use script, paste this command to the terminal: "chmod 700 start.sh"

If for some other reason the script still doesn't work, you can run the frontend and the backend with the console from the workspace directory:

frontend: cd frontend -> npm run dev

run a new terminal in workspace directory (you can use docker exec -it <name of the active container> bash)

backend: cd backend -> python3 manage.py runserver 0.0.0.0:8000

## When the application is running:
The interactive part of the application is running on the http://localhost:5173/ of your machine. The link should be displayed after running the application with the script or manually starting the frontend.

To stop the application use CTRL+C on the terminal(s) or simply stop the container.

To find out how to use the application, check the INSTRUCTIONS.md file

---
# Development environment
- The application is developed in a docker container in order to ensure compatibility between team members as well as make the installation of the application easier.
- For the machine learning part we're using python and its scikit-learn library, as well as pandas and numpy for data analysis and editing.
- The GUI visualisation will be done as a web application with a python backend using django framework, and Vue.js frontend and basic tools for development and build (curl, git etc.). For the full list of installed packages/libraries one can check reqirements.txt, frontend/package.json and the Dockerfile.
- Additional tools, that team members use for specific tasks are not included in this summary, but if a deliverable will be made using those tools, information about it will be included with it.

---

# Git repository structure:
- The **main** branch Contains all deliverables and is the final state of the project at the end of a sprint.
- Other branches are created at will by team members, and are used only for testing, if necessary.
- **data-cleaning** branch, as the name suggests, contains the pipeline for data cleaning (results are stored in the main branch)

## Directory contents

- **📂 models**
  - Contains ML model training scripts, dataset distribution, anomaly definitions, and documentation of model tuning processes.
- **📂 sprint_backlogs**
  - Contains all sprint documentation.
- **📂 backend**
  - Contains all files for the django backend application.
- **📂 frontend**
  - Contains all files for the Vue.js frontend application.
- **📂 .devcontainer**
  - Docker
- **INSTRUCTIONS.md**
  - Instructions on how to use the application

- The .md files in the repository are used as additional descriptions for specific parts - their titles indicate what they're about.

