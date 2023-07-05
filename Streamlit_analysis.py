import io
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

def calculate_fold_change(temp_df, genotype_col, control_group, var_idx):
    '''
    Calculates fold change values for different groups compared to a control group.
    Args:
        temp_df (pandas.DataFrame): Input DataFrame containing the data.
        genotype_col (str): Name of the column in `temp_df` representing the genotype.
        control_group (str): Name of the control group to compare against.
        var_idx (tuple): A tuple specifying the range of indices to consider in the DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame containing the fold change values.
    '''
    # Get the data that we want to use for the dataframe
    temp_df = temp_df.dropna()
    #var_idx = np.arange(var_idx[0], var_idx[1])
    genotypes = img_df[genotype_col].unique()
    # Calculate the mean values for the control group and save it into a new dataframe
    control_mean = np.mean(temp_df[(temp_df[genotype_col] == control_group)].iloc[:,var_idx], axis=0)
    fold_df = pd.DataFrame()
    fold_df['Features'] = temp_df.columns[var_idx]
    fold_df['Group'] = [control_group] * len(fold_df)
    fold_df['Fold Change'] = control_mean.values
    fold_df['p-value'] = control_mean.values * 0
    # Now loop through the genotypes to calculate the fold-change
    for gen in genotypes:
        if gen != control_group:
            # Step 1: calculate the fold change
            temp_fold_change = np.mean(temp_df[(temp_df[genotype_col] == gen)].iloc[:,var_idx], axis=0) / control_mean
            # Step 2: Perform statistical tests, adjust for multiple testing
            _, p_values = ttest_ind(temp_df[(temp_df[genotype_col] == gen)].iloc[:,var_idx], temp_df[(temp_df[genotype_col] == control_group)].iloc[:,var_idx])
            adjusted_p_values_1 = multipletests(p_values, method='fdr_bh')[1]
            # Step 3: Store the data into the fold_df
            gen_df = pd.DataFrame()
            gen_df['Features'] = temp_df.columns[var_idx]
            gen_df['Group'] = [gen] * len(gen_df)
            gen_df['Fold Change'] = np.log2(temp_fold_change.values)
            gen_df['p-value'] = -np.log10(p_values)
            fold_df = pd.concat([fold_df, gen_df])
    
    st.session_state.groupped_data = fold_df


def convert_df(temp_df):
    return temp_df.to_csv().encode('utf-8')



# Initialize the session
if "new_session" not in st.session_state:
    st.session_state["new_session"] = True
    st.session_state["multiple_group"] = False
    st.session_state.img_df = []
    st.session_state.groupped_data = []
    st.session_state.xMax = 0
    st.session_state.yMax = 0
    st.session_state.biplot = True
    st.session_state.lda_feature_list = False

varIdx = [8, 93]

# Create the basic layout of the page
st.title('Multidemensional analysis of Calcium Imaging Data')
with st.sidebar:
    add_input = st.file_uploader("Choose a *.csv file with the data you would like to analyze.", type=["csv"])
    if len(st.session_state.img_df) > 0:
        var_names = [var for var in st.session_state.img_df.keys()]
        options = st.multiselect('Select informative columns', var_names)
        varIdx = st.session_state.img_df.columns.isin(options)
        st.checkbox("Add more grouping factors?", key="multiple_group") 



if add_input is not None:
    st.session_state.img_df = pd.read_csv(add_input)

# Inspect the raw data
st.subheader('Raw data')

if len(st.session_state.img_df) > 0:
    img_df = st.session_state.img_df
    st.write(img_df)
    # Get the variables names and ask the user to pick the important ones
    var_names = [var for var in img_df.keys()]
    col1, col2, col3 = st.columns(3)
    control_group = []
    selection_group1 = []
    selection_group2 = []
    with col1:
        genotype_col = st.selectbox("Select the condition variable", var_names, key="sel_genotype")
        genotypes = img_df[genotype_col].unique()
        if genotype_col != "CellID":
            control_group = st.radio("Select the control condition", options=genotypes, key="radio_genotype")
    with col2:
        group1_col = st.selectbox("Select the condition variable", var_names, key="sel_group1", disabled= not st.session_state.multiple_group)
        if group1_col != "CellID":
            group1 = img_df[group1_col].unique()
            selection_group1 = st.radio("Select the grouping", options=group1, key="radio_group1", disabled= not st.session_state.multiple_group)
    with col3:
        group2_col = st.selectbox("Select the condition variable", var_names, key="sel_group2", disabled= not st.session_state.multiple_group)
        if group2_col != "CellID":
            group2 = img_df[group2_col].unique()
            selection_group2 = st.radio("Select the grouping", options=group2, key="radio_group2", disabled= not st.session_state.multiple_group)
    
    use_df = img_df.copy()
    if len(selection_group1) != 0:
        use_df = img_df.loc[img_df[group1_col]==selection_group1,:]
    if len(selection_group2) != 0:
        use_df = use_df[(use_df[group2_col]==selection_group2)]
    st.button("Calculate the fold change", key="fold_change", on_click=calculate_fold_change, args=(use_df, genotype_col, control_group, varIdx))

    if len(st.session_state.groupped_data) > 0:
        st.subheader('Fold change')
        st.write(st.session_state.groupped_data)
        download_df = convert_df(st.session_state.groupped_data)
        st.download_button(label="Download data", data=download_df, file_name="NetworkAnalysis_FoldChange.csv", mime="text/csv")


        # Plot the data indicating the significant threshold as a dotted line
        st.subheader('Volcano plot')
        tab1, tab2 = st.tabs(["Seaborn plot (default)", "Vega-lite plot (experimental)"])
        with tab1:
            cmap=['#252525', '#026842', '#5E9BD1']
            xlim = [-2, 2]
            ylim = [0, 9]
            fig_volcano = plt.figure()
            sns.scatterplot(data=st.session_state.groupped_data, x='Fold Change', y='p-value', hue='Group', palette=cmap)
            plt.axhline(-np.log10(0.05), color='red', linestyle='dotted')
            plt.axvline(-1, color='black', linestyle='dotted')
            plt.axvline(1, color='black', linestyle='dotted')
            if st.session_state.xMax > 0.0:
                plt.xlim(np.array([-1, 1])*st.session_state.xMax)
            if st.session_state.yMax > 0.0:
                plt.ylim([0, st.session_state.yMax])
            st.pyplot(fig_volcano, use_container_width=True)
            img_volcano = io.BytesIO()
            plt.savefig(img_volcano, format='pdf')
            col1, col2, col3, col4 = st.columns(4)
            st.session_state.xMax = int(col1.number_input('X-axis limit'))
            st.session_state.yMax = int(col2.number_input('Y-axis limit'))
            col3.write("##"); col3.button("Relead")
            col4.write("##"); col4.download_button(label="Download volcano", data=img_volcano, file_name="volcanoPlot.pdf", mime="application/pdf")
            st.write('''Some explanation of the way the volcano plot is plotted.  
                        For example what is the red dotted line and the two black dotted lines.
            ''')

        with tab2:
            # Plot using vega
            chart = {"mark": "point",
                    "encoding": {
                        "x": {"field": "Fold Change", "type": "quantitative", },
                        "y": {"field": "p-value", "type": "quantitative", },
                        "color": {"field": "Group", "type": "nominal"},
                        },
                    }
            st.vega_lite_chart(st.session_state.groupped_data, chart, theme="streamlit", use_container_width=True)


        st.subheader('LDA analysis')
        # Perform an LDA
        lda_data = use_df.iloc[:,varIdx]
        lda_labels = use_df.loc[:,genotype_col]
        # Returns a scaled version of your sliced data 
        scaler = preprocessing.StandardScaler()
        lda_data = scaler.fit_transform(lda_data)
        lda_data = np.nan_to_num(lda_data, nan=0.0001)
        # Input the column you want to use as the labels for PCA: make sure to take data.values. 
        #le = LabelEncoder()
        #labels = le.fit_transform(temp_df['Genotype'].values)
        # set lda model and fit to the data
        linear_discriminant = LDA()
        lda = linear_discriminant.fit_transform(lda_data, lda_labels)
        # Get the coefficients (scaling factors)
        lda_coef = linear_discriminant.scalings_
        feat_names = use_df.iloc[:,varIdx].columns.tolist()
        sum_components = np.sum(abs(lda_coef), axis=1)
        sort_index = np.argsort(sum_components)
        scalex = 1.0 / (lda[:, 0].max() - lda[:, 0].min())
        scaley = 1.0 / (lda[:, 1].max() - lda[:, 1].min())

        fig_lda = plt.figure()
        if st.session_state.biplot:
            sns.scatterplot(x=lda[:, 0] * scalex, y=lda[:, 1] * scaley, hue=lda_labels, palette=cmap)
            for s in sort_index[0:5]:
                plt.arrow(0, 0, lda_coef[s, 0], lda_coef[s, 1], color='r')
                plt.text(lda_coef[s, 0], lda_coef[s, 1], feat_names[s], ha='center', va='center')
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
        else:
            sns.scatterplot(x=lda[:, 0], y=lda[:, 1], hue=lda_labels, palette=cmap)
        plt.xlabel("LD1")
        plt.ylabel("LD2")
        st.pyplot(fig_lda, use_container_width=True)
        img_lda = io.BytesIO()
        plt.savefig(img_lda, format='pdf')
        col1, col2, col3, col4 = st.columns(4)
        st.session_state.biplot = col1.checkbox("Biplot", value=st.session_state.biplot, key="lda_biplot")
        col3.button("Reload")
        lda_feature_list = col2.checkbox("Show list of features", value=st.session_state.lda_feature_list, key="lda_feature_list")
        col4.download_button(label="Download LDA plot", data=img_lda, file_name="LDA_Plot.pdf", mime="application/pdf")
        if lda_feature_list:
            for s in sort_index:
                feat_names[s]
            #ordered_names = [feat_names[s] for s in sort_index]
            #ordered_names
    