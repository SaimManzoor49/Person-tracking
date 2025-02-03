# Project Setup

This guide will help you set up and run the project on your local machine.

## Prerequisites

- Python 3.8 or later installed on your system.  
- Pip (Python package installer) installed.

---

## Step 1: Set Up a Virtual Environment

Creating and activating a virtual environment ensures dependencies are managed cleanly. Follow the steps for your operating system:

### Windows
1. Open Command Prompt or PowerShell.
2. Run the following commands:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

### Mac/Linux
1. Open a terminal.
2. Run the following commands:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

---

## Step 2: Install Dependencies

1. Ensure the virtual environment is activated (`venv`).
2. Run the following command to install all required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Step 3: Configure Models or Video Sources

1. Open the configuration files or main file.
2. Adjust the paths to models, video sources, or other settings as required for your use case.

---

## Step 4: Run the Project

After completing the above steps, you can start the project as needed. Ensure your environment is active during runtime.
 ```cmd
  python main.py
   ```
---

## Notes

- If you encounter any issues, double-check that the virtual environment is activated and dependencies are installed.
- For specific configuration changes, refer to the inline comments in the scripts.

---

Enjoy using the project! If you have any questions or need assistance, feel free to ask.
