# PartialCorWood
# Wood Physical Properties: Exact Partial Correlation Analysis

**Authors:** Julia Sonsin-Oliveira and Deborah Bambil

## Overview
This repository provides a robust Python script designed to calculate the exact partial correlation between anatomical features and a target physical property, while mathematically controlling for a confounding variable.

This tool is essential for disentangling interdependencies in wood physics (e.g., isolating the pure effect of tissue arrangement on Volumetric Shrinkage by partialling out the variance shared with Wood Density). To handle the unit sum constraint, it automatically applies the Centred Log-Ratio (CLR) transformation to sub-compositional data and drops baselines to prevent perfect structural collinearity.

## Dependencies
The script is equipped with an automatic dependency installer. Upon the first run, it will automatically check for and install the required libraries. The primary dependencies are:
* `pandas` & `numpy` (Data manipulation)
* `scipy` (Statistical calculations for exact partial correlation and p-values)
* `scikit-bio` (CLR transformation for compositional data)
* `scikit-learn` (Data standardization via Z-score)
* `matplotlib` & `seaborn` (Data visualization)

## How to Use

1. **Prepare your data:** Ensure your dataset is in `.csv` or `.txt` format. Variables should be in columns and samples (species/specimens) in rows. 
2. **Run the script:** Execute the Python file (`.py`) in your terminal or IDE.
3. **Follow the prompts:** 
   * The script features a Universal File Reader. Simply type the name of your file (e.g., `my_data.csv`) when prompted.
   * You will be asked to manually input the exact names of your **Target Variable** (e.g., VS) and your **Control Variable** (e.g., WD).
4. **Retrieve Results:** The script will automatically export a detailed CSV table with exact *r* and *p*-values, and generate a high-resolution bar plot highlighting significant positive and negative correlations in the same directory.

## License
This project is open-source and available for the scientific community. If you utilize this script in your research, please cite the associated publication.
