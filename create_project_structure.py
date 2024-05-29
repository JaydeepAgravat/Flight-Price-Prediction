from pathlib import Path

def create_project_structure():
    # Define the directory structure
    directories = [
        "data",
        "notebooks",
        "models",
        "src/scripts",  
        "src/utils"
    ]

    # Create the directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Create the empty files
    files = [
        "src/scripts/__init__.py",
        "src/scripts/data_cleaning.py",
        "src/scripts/model_training.py",
        "src/scripts/model_evaluation.py",
        "src/scripts/model_save.py",
        "src/utils/__init__.py",
        "src/utils/utils.py",
        
        "requirements.txt",
        "README.md",
        ".gitignore"
        "main.py"
    ]

    # Create empty files
    for file in files:
        Path(file).touch()

    print("Directory structure created successfully.")

def main():
    create_project_structure()

if __name__ == "__main__":
    main()
