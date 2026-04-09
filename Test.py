import streamlit as st
import numpy as np
import pandas as pd

import math

st.title("Hypothesis Testing Tool")

# --- Helper function ---
def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

# --- Sidebar ---
st.sidebar.header("Data Input")
data_option = st.sidebar.selectbox("How to input data?", ["Manual Input", "Upload CSV"])

hypothesis_statement = ""

if data_option == "Upload CSV":
    hypothesis_statement = st.sidebar.text_area("Describe your hypothesis (optional)", "")

if data_option == "Manual Input":
    st.sidebar.subheader("Group 1 Data")
    group1_input = st.sidebar.text_area("Enter values for Group 1 (comma-separated)", "1,2,3,4,5")
    group1 = [float(x.strip()) for x in group1_input.split(',') if x.strip()]

    st.sidebar.subheader("Group 2 Data")
    group2_input = st.sidebar.text_area("Enter values for Group 2 (comma-separated)", "6,7,8,9,10")
    group2 = [float(x.strip()) for x in group2_input.split(',') if x.strip()]

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if len(df.columns) >= 2:
            group1 = df.iloc[:, 0].dropna().tolist()
            group2 = df.iloc[:, 1].dropna().tolist()
        else:
            st.error("CSV must have at least 2 columns")
            st.stop()
    else:
        st.stop()

# --- Settings ---
st.sidebar.header("Test Settings")
test_type = st.sidebar.selectbox("Select Test Type", ["T-Test", "Z-Test"])
test_variant = st.sidebar.selectbox("Test Variant", ["Two-sample", "One-sample (vs mean)"])

if test_variant == "One-sample (vs mean)":
    null_mean = st.sidebar.number_input("Null hypothesis mean", value=0.0)
else:
    null_mean = None

hypothesis_type = st.sidebar.selectbox(
    "Hypothesis Type",
    ["Two-sided", "Left-tailed (Group1 < Group2)", "Right-tailed (Group1 > Group2)"]
)

alternative = {
    "Two-sided": "two-sided",
    "Left-tailed (Group1 < Group2)": "less",
    "Right-tailed (Group1 > Group2)": "greater"
}[hypothesis_type]

alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)

st.sidebar.header("Assumptions")
normality = st.sidebar.checkbox("Data is normally distributed", value=True)
equal_var = st.sidebar.checkbox("Variances are equal", value=True)
independence = st.sidebar.checkbox("Samples are independent", value=True)

# --- Run Test ---
if st.button("Run Test"):

    if not group1 or not group2:
        st.error("Please provide data for both groups")
        st.stop()

    # Summary
    st.header("Data Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Group 1")
        st.write(f"Mean: {np.mean(group1):.4f}")
        st.write(f"Std: {np.std(group1, ddof=1):.4f}")
        st.write(f"N: {len(group1)}")

    with col2:
        st.subheader("Group 2")
        st.write(f"Mean: {np.mean(group2):.4f}")
        st.write(f"Std: {np.std(group2, ddof=1):.4f}")
        st.write(f"N: {len(group2)}")

    # Assumptions
    st.header("Assumptions Check")
    assumptions_ok = normality and equal_var and independence

    if assumptions_ok:
        st.success("All assumptions accepted.")
    else:
        st.warning("Some assumptions may be violated.")

    # --- Test Calculation (NO SCIPY) ---
    if test_variant == "One-sample (vs mean)":
        data = group1 + group2
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)

        stat = (mean - null_mean) / (std / np.sqrt(n))

    else:
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)

        stat = (mean1 - mean2) / np.sqrt(std1**2/n1 + std2**2/n2)

    # P-value using normal approx
    if alternative == "two-sided":
        p_value = 2 * (1 - normal_cdf(abs(stat)))
    elif alternative == "less":
        p_value = normal_cdf(stat)
    else:
        p_value = 1 - normal_cdf(stat)

    # --- Results ---
    st.header("Test Results")
    st.write(f"Test Statistic: {stat:.4f}")
    st.write(f"P-value: {p_value:.4f}")

    if p_value < alpha:
        st.success("Reject Null Hypothesis")
    else:
        st.info("Fail to Reject Null Hypothesis")

    # --- Power (simple approx) ---
    st.header("Power (Approx)")
    try:
        effect_size = abs(np.mean(group1) - np.mean(group2)) / np.std(group1 + group2)
        power = min(1.0, effect_size * np.sqrt(len(group1)))
        st.write(f"Approx Power: {power:.3f}")
    except:
        st.write("Power not available")

    # --- Plot ---
    st.header("Visualization")
    fig, ax = plt.subplots()
    ax.boxplot([group1, group2], labels=["Group 1", "Group 2"])
    st.pyplot(fig)
