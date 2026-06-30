# Image Processing Project (COMP2032)

This repository contains the Image Processing project for the Spring 2023-2024 Semester, part of the COMP2032 module. The project demonstrates various image processing techniques using Python.

## Table of Contents
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Setting Up a Virtual Environment](#setting-up-a-virtual-environment)
  * [Installing Dependencies](#installing-dependencies)
* [Running the Project](#running-the-project)
* [License](#license)

## Getting Started

These instructions will guide you through setting up your project environment and running the project on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have Python installed on your system. This project is tested with Python 3.11.4+, but it should work with any Python 3.x.x version. You can download Python from [https://www.python.org/](https://www.python.org/).

### Setting Up a Virtual Environment

A virtual environment is a self-contained directory tree that contains a Python installation for a particular version of Python, plus a number of additional packages. Using a virtual environment avoids conflicts between project dependencies.

To set up a virtual environment, follow these steps:

1. Open a terminal or command prompt.
2. Navigate to the project's root directory.
3. Run the following command to create a virtual environment named `venv`:

   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment:

   - On Windows, run:

   ```cmd
   .\venv\Scripts\activate
   ```

   - On Unix or MacOS, run:

   ```bash
   source venv/bin/activate
   ```

After activation, your command prompt will change to show the name of the activated environment.

### Installing Dependencies

With the virtual environment activated, install the project dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This command reads the `requirements.txt` file in your project directory and installs all the libraries listed in it.

To exit the virtual enviromnent, use the following command:

```bash
deactivate
```

## Running the Project

After installing the dependencies, you're ready to run the project. To start the project, use the following command (make sure your virtual environment is activated):

```bash
python src/gui.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
