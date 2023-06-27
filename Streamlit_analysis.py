import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 'truetype'
import seaborn as sns
sns.set_theme(style='ticks', rc={'axes.spines.right':False, 'axes.spines.top':False})
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# preprocessing
from sklearn import metrics, preprocessing
from sklearn.preprocessing import LabelEncoder, scale

# classifiers
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# models
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# training
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# metrics
from sklearn.metrics import confusion_matrix, accuracy_score


def plot_LDA_biplot(lda, lda_labels, lda_coef, colors, feat_names, ax=None):
    sum_components = np.sum(abs(lda_coef), axis=1)
    sort_index = np.argsort(sum_components)
    scalex = 1.0 / (lda[:, 0].max() - lda[:, 0].min())
    scaley = 1.0 / (lda[:, 1].max() - lda[:, 1].min())

    if ax is None:
        ax = plt.gca()

    sns.scatterplot(x=lda[:, 0] * scalex, y=lda[:, 1] * scaley, hue=lda_labels, palette=colors, ax=ax)
    for s in sort_index[0:5]:
        ax.arrow(0, 0, lda_coef[s, 0], lda_coef[s, 1], color='r')
        ax.text(lda_coef[s, 0] * 1.5, lda_coef[s, 1] * 1.5, feat_names[s], ha='center', va='center')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    
    return ax, sort_index



# Create the basic layout of the page
st.title('Multidemensional analysis of Calcium Imaging Data')
with st.sidebar:
    add_input = st.file_uploader("Choose a *.csv file with the data you would like to analyze.", type=["csv"])

img_df = []
if add_input is not None:
    img_df = pd.read_csv(add_input)

# Load the data, for now use this path, then replace with a load button
#img_df = pd.read_csv("G:/My Drive/UCB/Patient_Calcium/EINetwork_complete.csv")

# Inspect the raw data
st.subheader('Raw data')
if len(img_df) == 0:
    st.write('Here is were we can see the raw data!')
else:
    st.write(img_df)
    # Get the variables names and ask the user to pick the important ones
    var_names = [var for var in img_df.keys()]
    st.write(var_names)
    col1, col2, col3 = st.columns(3)
    with col1:
        genotype_col = st.selectbox("Select the condition variable", var_names)
        st.write("You selected: ", genotype_col)

    
    # Calculate the fold change and plot it as a volcano plot
    varIdx = np.arange(8, 93)
    control_data = img_df[(img_df['Genotype'] == 'Control') & (img_df['LVV']=='Sham') & (img_df['RecID']=='WIC4')].iloc[:, varIdx]
    d207_data = img_df[(img_df['Genotype'] == 'D207G') & (img_df['LVV']=='Sham') & (img_df['RecID']=='WIC4')].iloc[:, varIdx]
    s241_data = img_df[(img_df['Genotype'] == 'S241fs') & (img_df['LVV']=='Sham') & (img_df['RecID']=='WIC4')].iloc[:, varIdx]
    # Drop nan values (if any)
    control_data = control_data.dropna()
    d207_data = d207_data.dropna()
    s241_data = s241_data.dropna()
    # Step 1: Calculate fold change
    fold_change_1 = np.mean(d207_data, axis=0) / np.mean(control_data, axis=0)
    fold_change_2 = np.mean(s241_data, axis=0) / np.mean(control_data, axis=0)
    # Step 2: Perform statistical tests, adjust for multiple testing
    _, p_values_1 = ttest_ind(d207_data, control_data)
    _, p_values_2 = ttest_ind(s241_data, control_data)
    adjusted_p_values_1 = multipletests(p_values_1, method='fdr_bh')[1]
    adjusted_p_values_2 = multipletests(p_values_2, method='fdr_bh')[1]
    # Step 3: Collect the data into a new dataframe
    fold_df = pd.DataFrame()
    fold_df['Features'] = img_df.columns[varIdx]
    fold_df['Group'] = ['D207G'] * len(fold_df)
    fold_df['Fold Change'] = np.log2(fold_change_1.values)
    fold_df['p-value'] = -np.log10(p_values_1)
    temp_df = pd.DataFrame()
    temp_df['Features'] = img_df.columns[varIdx]
    temp_df['Group'] = ['S241fs'] * len(temp_df)
    temp_df['Fold Change'] = np.log2(fold_change_2.values)
    temp_df['p-value'] = -np.log10(p_values_2)
    fold_df = pd.concat([fold_df, temp_df])

    # Show the data
    st.subheader('Fold change')
    st.write(fold_df)


    # Plot the data indicating the significant threshold as a dotted line
    st.subheader('Volcano plot')
    colors=['#026842', '#5E9BD1']
    fig = plt.figure()
    sns.scatterplot(data=fold_df, x='Fold Change', y='p-value', hue='Group', palette=colors)
    plt.axhline(-np.log10(0.01), color='red', linestyle='dotted')
    plt.axvline(-1, color='black', linestyle='dotted')
    plt.axvline(1, color='black', linestyle='dotted')
    plt.xlim(-2,2)
    plt.ylim(0,9)
    st.pyplot(fig, use_container_width=True)
    