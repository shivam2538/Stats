import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.power import tt_ind_solve_power, zt_ind_solve_power

st.title("Hypothesis Testing Tool")

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

st.sidebar.header("Test Settings")
test_type = st.sidebar.selectbox("Select Test Type", ["T-Test", "Z-Test"])
test_variant = st.sidebar.selectbox("Test Variant", ["Two-sample", "One-sample (vs mean)"])

if test_variant == "One-sample (vs mean)":
    null_mean = st.sidebar.number_input("Null hypothesis mean", value=0.0)
else:
    null_mean = None

hypothesis_type = st.sidebar.selectbox("Hypothesis Type", ["Two-sided", "Left-tailed (Group1 < Group2)", "Right-tailed (Group1 > Group2)"])

if hypothesis_type == "Two-sided":
    alternative = 'two-sided'
elif hypothesis_type == "Left-tailed (Group1 < Group2)":
    alternative = 'less'
else:
    alternative = 'greater'

alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)

st.sidebar.header("Assumptions")
normality = st.sidebar.checkbox("Data is normally distributed", value=True)
equal_var = st.sidebar.checkbox("Variances are equal", value=True)
independence = st.sidebar.checkbox("Samples are independent", value=True)

if st.button("Run Test"):
    if not group1 or not group2:
        st.error("Please provide data for both groups")
    else:
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

        st.header("Assumptions Check")
        assumptions_ok = normality and equal_var and independence
        if assumptions_ok:
            st.success("All assumptions accepted.")
        else:
            st.warning("Some assumptions may be violated. Results should be interpreted cautiously.")

        st.header("Hypotheses")
        if hypothesis_statement:
            st.write(f"Custom Hypothesis Statement: {hypothesis_statement}")
        if test_variant == "One-sample (vs mean)":
            if alternative == 'two-sided':
                st.write(f"Null Hypothesis (H₀): Mean = {null_mean}")
                st.write(f"Alternative Hypothesis (H₁): Mean ≠ {null_mean}")
            elif alternative == 'less':
                st.write(f"Null Hypothesis (H₀): Mean ≥ {null_mean}")
                st.write(f"Alternative Hypothesis (H₁): Mean < {null_mean}")
            else:
                st.write(f"Null Hypothesis (H₀): Mean ≤ {null_mean}")
                st.write(f"Alternative Hypothesis (H₁): Mean > {null_mean}")
        else:
            if alternative == 'two-sided':
                st.write("Null Hypothesis (H₀): Mean₁ = Mean₂")
                st.write("Alternative Hypothesis (H₁): Mean₁ ≠ Mean₂")
            elif alternative == 'less':
                st.write("Null Hypothesis (H₀): Mean₁ ≥ Mean₂")
                st.write("Alternative Hypothesis (H₁): Mean₁ < Mean₂")
            else:
                st.write("Null Hypothesis (H₀): Mean₁ ≤ Mean₂")
                st.write("Alternative Hypothesis (H₁): Mean₁ > Mean₂")

        # Perform test
        if test_type == "T-Test":
            if test_variant == "One-sample (vs mean)":
                t_stat, p_value = stats.ttest_1samp(group1 + group2, null_mean, alternative=alternative)
            else:
                t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var, alternative=alternative)
        else:  # Z-Test
            if test_variant == "One-sample (vs mean)":
                combined = group1 + group2
                sample_mean = np.mean(combined)
                sample_std = np.std(combined, ddof=1)
                n = len(combined)
                se = sample_std / np.sqrt(n)
                z_stat = (sample_mean - null_mean) / se
                if alternative == 'two-sided':
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                elif alternative == 'less':
                    p_value = stats.norm.cdf(z_stat)
                else:
                    p_value = 1 - stats.norm.cdf(z_stat)
            else:
                mean1, mean2 = np.mean(group1), np.mean(group2)
                std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
                n1, n2 = len(group1), len(group2)
                se = np.sqrt(std1**2/n1 + std2**2/n2)
                z_stat = (mean1 - mean2) / se
                if alternative == 'two-sided':
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                elif alternative == 'less':
                    p_value = stats.norm.cdf(z_stat)
                else:
                    p_value = 1 - stats.norm.cdf(z_stat)

        st.header("Test Results")
        if test_type == "T-Test":
            st.write(f"T-statistic: {t_stat:.4f}")
        else:
            st.write(f"Z-statistic: {z_stat:.4f}")
        st.write(f"P-value: {p_value:.4f}")
        st.write(f"Significance level (α): {alpha}")

        if p_value < alpha:
            st.success("Decision: Reject the null hypothesis.")
            st.write("Conclusion: There is a significant difference in the specified direction.")
            if test_variant == "Two-sample":
                mean1, mean2 = np.mean(group1), np.mean(group2)
                if alternative == 'less':
                    st.write(f"Group 1 has significantly lower mean ({mean1:.2f} vs {mean2:.2f})")
                elif alternative == 'greater':
                    st.write(f"Group 1 has significantly higher mean ({mean1:.2f} vs {mean2:.2f})")
                else:
                    if mean1 > mean2:
                        st.write(f"Group 1 has higher mean ({mean1:.2f} vs {mean2:.2f})")
                    else:
                        st.write(f"Group 2 has higher mean ({mean2:.2f} vs {mean1:.2f})")
        else:
            st.info("Decision: Fail to reject the null hypothesis.")
            st.write("Conclusion: No significant difference detected in the specified direction.")

        st.header("Evidence Against Null Hypothesis")
        st.write("The p-value represents the probability of observing the data (or more extreme) assuming the null hypothesis is true.")
        if p_value < 0.001:
            st.write("**Very strong evidence** against the null hypothesis (p < 0.001).")
        elif p_value < 0.01:
            st.write("**Strong evidence** against the null hypothesis (p < 0.01).")
        elif p_value < 0.05:
            st.write("**Moderate evidence** against the null hypothesis (p < 0.05).")
        elif p_value < 0.10:
            st.write("**Weak evidence** against the null hypothesis (p < 0.10).")
        else:
            st.write("**No evidence** against the null hypothesis (p ≥ 0.10).")

        st.header("Error Rates")
        st.write(f"Type I Error Rate (False Positive): {alpha}")
        if assumptions_ok and test_variant == "Two-sample":
            # Calculate power
            effect_size = abs(np.mean(group1) - np.mean(group2)) / np.sqrt((np.var(group1) + np.var(group2))/2)
            if test_type == "T-Test":
                power = tt_ind_solve_power(effect_size=effect_size, nobs1=len(group1), alpha=alpha, ratio=len(group2)/len(group1), alternative='two-sided')
            else:
                power = zt_ind_solve_power(effect_size=effect_size, nobs1=len(group1), alpha=alpha, ratio=len(group2)/len(group2), alternative='two-sided')
            st.write(f"Estimated Power (1 - Type II Error): {power:.4f}")
        else:
            st.write("Power calculation requires valid assumptions and two-sample test.")

        # Plot
        st.header("Data Visualization")
        fig, ax = plt.subplots()
        ax.boxplot([group1, group2], labels=['Group 1', 'Group 2'])
        ax.set_title('Box Plot of Groups')
        st.pyplot(fig)
        ax.set_ylabel('Values')
        ax.grid(True)
        st.pyplot(fig)
