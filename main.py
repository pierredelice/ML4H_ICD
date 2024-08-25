# Import the necessary functions from your_script.py
from modules import install_and_import, is_conda, parse_requirements, freeze_requirements, main
from src.data.data_loader import read_data

def main_script():
    # Check if conda is available
    conda = is_conda()
    print(f"Is conda available? {conda}")

    # Install a specific package if it's not already installed
    install_and_import('numpy')

    # Parse a requirements.txt file
    requirements = parse_requirements('requirements.txt')
    print(f"Parsed requirements: {requirements}")

    # Freeze current environment packages to requirements.txt
    freeze_requirements('frozen_requirements.txt')
    print("Requirements have been frozen to frozen_requirements.txt")

    # Read data
    path = 'Data/icd_clean.pkl'
    print(read_data(path))

    # Run the main function from your_script.py
    main()

if __name__ == "__main__":
    main_script()

