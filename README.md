### Demonstration of Antenna Autotuning by Machine Learning

# Introduction
This is a sub-project for antenna design automation. To nearly fully automate the antenna design process subject to basic antenna parameters specifications, please refer to my [main project](https://github.com/electronics10/Topology_Optimization). The computational power and effort are unreal for individual researchers to train a good enough model to replace electromagnetic solver. Thus, work and effort exerted to automate the antenna design by machine learning often result in unuseful or redundant small models that only applies to very specific problems. With this being said, I decided to focus on autotuning PIFA antenna that operates at 2.45 GHz. The reason is that PIFA antenna is an easy yet great and commonly used antenna design. Although PIFA is such a convenient and well studied antenna design, the surrounding materials, e.g. the cover or frame of a phone and the feed point postion, may change in different product designs. Therefore, a few proff of concepts models that can autotune the PIFA antenna parameters according to the surroundings are trained in this project to demonstrate the integrationg of ML in antenna design.

A tabular data `data\data.csv` would be used to train the model. The dataset is acquired by utilizing CST Studio Suite® optimizer to make sure the parameters setup of all PIFA antenna have S11 from 2.3~2.6 GHz below -10 dB. The acqusition process is automated by the script `pre_data_acqusition.py`, and the software setup is identical to my [main project](https://github.com/electronics10/Topology_Optimization). One can obtain their own data by modifying and running the script. The dataset is in the form of 13 numbers in each row/sample with `500` samples. The first ten numbers are the input, with the first number representing the x postion of feed point and the remaining represents a binary sequence implying whether there are PEC surrounding blocks at some fixed positions. The rest 3 numbers are the output, representing the parameters of PIFA antenna shape. A few models are trained on the data and can be compared parallely. To justify the comparison, hyperparameter optimization for each model is performed using random search with 5-fold cross-validation, evaluating 15 trials per model. The best configurations were used to train the final models, ensuring a fair comparison.

## Prerequisites
Before running the code, ensure you have the following installed:
- **Git**: To clone this repository.
- **Miniconda** or **Anaconda**: To manage the Python environment and dependencies.

---

## Step-by-Step Setup Instructions

### 1. Install Git
- **Windows**: Download and install Git from [git-scm.com](https://git-scm.com/downloads). Follow the default installation options.
- **Mac**: Open a terminal and run `brew install git` (if you have Homebrew) or download from the website above.
- **Linux**: Open a terminal and run `sudo apt install git` (Ubuntu/Debian) or `sudo yum install git` (CentOS/RHEL).

Verify Git is installed by running:
```
git --version
```

---

### 2. Install Miniconda
Miniconda is a lightweight version of Anaconda that manages Python environments.
- Download Miniconda from [docs.conda.io](https://docs.conda.io/en/latest/miniconda.html) (choose the version for your OS: Windows, Mac, or Linux).
- Follow the installer instructions:
  - **Windows**: Double-click the `.exe` file and follow the prompts. Check "Add Miniconda to my PATH" if available.
  - **Mac/Linux**: Run the downloaded `.sh` file in a terminal (e.g., `bash Miniconda3-latest-Linux-x86_64.sh`) and follow the prompts.
- Restart your terminal after installation.

Verify Miniconda is installed by running:
```
conda --version
```

---

### 3. Clone the Repository
In a terminal or command prompt, navigate to the folder where you want the project and run:
```
git clone https://github.com/electronics10/mlpifa.git
```
Then, enter the project directory:
```
cd mlpifa
```

---

### 4. Set Up the Conda Environment

This project uses a predefined environment file (environment.yml) to install all dependencies.

- Create the environment:
  - **Windows**:
  ```
  conda env create -f environment_win.yml
  ```

  - **Mac**:
  ```
  conda env create -f environment_mac.yml
  ```

- Activate the environment:
```
conda activate mlpifa
```

Please refer to my [main project](https://github.com/electronics10/Topology_Optimization) to create environment `autotune`.

---

### 5. Run the Code
- For new or more data acquirement:
  ```
  conda activate autotune
  python pre_data_acquisition.py
  ```

- After acquiring the data, filtered bad data samples:
  ```
  python pre_filter_data.py
  ```

- For training, change the the name of the filtered data `data_filtered.csv` to `data.csv` and run the script:
  ```
  conda deactivate
  conda activate mlpifa
  python train_FNN.py
  ```
  Trained models would be saved to `./artifacts_FNN`. Training and validation loss and Prediction vs. True value figures would also be stored. Most importantly, we'll get a file `post_prediction.csv` for further inspection or post processing with 15 randomly genereted inputs and corresponding predictions by the trained model.

- To show and compare the results of the models in post processing, copy `post_prediction.csv` to the folder `./data` and run the scripts:
     ```
     conda deactivate
     conda activate autotune
     python post_check.py
     python post_compare.py
     python post_plot_s11.py
     ```
  `post_check.py` file is used to map the predictions to its closest upper bound satisfying the physical constraints of the design region. `post_compare.py` is used to obtain the s11 data by CST.
---

## Troubleshooting
- If `conda` commands don’t work, ensure Miniconda is added to your system PATH or restart your terminal.
- For errors during `conda env create`, ensure you have an active internet connection, as it downloads packages.

---
