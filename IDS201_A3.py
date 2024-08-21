# Python code to create plots for Assignment 3

from google.colab import files
uploaded = files.upload()

import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('passwords_dataset.csv')

# Distribution of Password Strength

plt.figure(figsize=(8, 6))
df['Strength'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Password Strength')
plt.xlabel('Strength')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Password Length Distribution
plt.figure(figsize=(8, 6))
df['Length'].plot(kind='hist', bins=range(8, 18), color='lightgreen', edgecolor='black')
plt.title('Password Length Distribution')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.show()

# Pie charts for character type inclusion
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

df['Has Lowercase'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axs[0], colors=['lightcoral', 'lightskyblue'])
axs[0].set_title('Lowercase Characters')
axs[0].set_ylabel('')

df['Has Uppercase'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axs[1], colors=['lightcoral', 'lightskyblue'])
axs[1].set_title('Uppercase Characters')
axs[1].set_ylabel('')

df['Has Special Character'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axs[2], colors=['lightcoral', 'lightskyblue'])
axs[2].set_title('Special Characters')
axs[2].set_ylabel('')

plt.show()

# Scatter plot for Strength vs Length
plt.figure(figsize=(8, 6))
colors = {'Weak': 'red', 'Medium': 'orange', 'Strong': 'green'}
plt.scatter(df['Length'], df['Strength'].map(colors))
plt.title('Strength vs Length')
plt.xlabel('Length')
plt.ylabel('Strength')
plt.show()


# Grouped bar charts for Character Types vs Strength
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

df.groupby(['Strength', 'Has Lowercase']).size().unstack().plot(kind='bar', ax=axs[0], color=['lightcoral', 'lightskyblue'])
axs[0].set_title('Lowercase Characters')
axs[0].set_xlabel('Strength')

df.groupby(['Strength', 'Has Uppercase']).size().unstack().plot(kind='bar', ax=axs[1], color=['lightcoral', 'lightskyblue'])
axs[1].set_title('Uppercase Characters')
axs[1].set_xlabel('Strength')

df.groupby(['Strength', 'Has Special Character']).size().unstack().plot(kind='bar', ax=axs[2], color=['lightcoral', 'lightskyblue'])
axs[2].set_title('Special Characters')
axs[2].set_xlabel('Strength')

plt.show()

import seaborn as sns

# Box Plot for Length vs Strength
plt.figure(figsize=(10, 6))
sns.boxplot(x='Strength', y='Length', data=df, palette='Set2')
plt.title('Password Length Distribution by Strength')
plt.xlabel('Strength')
plt.ylabel('Length')
plt.show()

# Bar Chart for Average Length by Strength
plt.figure(figsize=(10, 6))
average_lengths = df.groupby('Strength')['Length'].mean()
average_lengths.plot(kind='bar', color='lightblue')
plt.title('Average Password Length by Strength')
plt.xlabel('Strength')
plt.ylabel('Average Length')
plt.xticks(rotation=0)
plt.show()

# Stacked Bar Chart for Character Types vs Strength
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Lowercase Characters
df_lowercase = df.groupby(['Strength', 'Has Lowercase']).size().unstack().fillna(0)
df_lowercase.div(df_lowercase.sum(1), axis=0).plot(kind='bar', stacked=True, ax=axs[0], color=['lightcoral', 'lightskyblue'])
axs[0].set_title('Lowercase Characters')
axs[0].set_xlabel('Strength')

# Uppercase Characters
df_uppercase = df.groupby(['Strength', 'Has Uppercase']).size().unstack().fillna(0)
df_uppercase.div(df_uppercase.sum(1), axis=0).plot(kind='bar', stacked=True, ax=axs[1], color=['lightcoral', 'lightskyblue'])
axs[1].set_title('Uppercase Characters')
axs[1].set_xlabel('Strength')

# Special Characters
df_special = df.groupby(['Strength', 'Has Special Character']).size().unstack().fillna(0)
df_special.div(df_special.sum(1), axis=0).plot(kind='bar', stacked=True, ax=axs[2], color=['lightcoral', 'lightskyblue'])
axs[2].set_title('Special Characters')
axs[2].set_xlabel('Strength')

plt.show()

# Grouped Bar Chart for Character Types vs Strength
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Lowercase Characters
df.groupby(['Strength', 'Has Lowercase']).size().unstack().plot(kind='bar', ax=axs[0], color=['lightcoral', 'lightskyblue'])
axs[0].set_title('Lowercase Characters')
axs[0].set_xlabel('Strength')

# Uppercase Characters
df.groupby(['Strength', 'Has Uppercase']).size().unstack().plot(kind='bar', ax=axs[1], color=['lightcoral', 'lightskyblue'])
axs[1].set_title('Uppercase Characters')
axs[1].set_xlabel('Strength')

# Special Characters
df.groupby(['Strength', 'Has Special Character']).size().unstack().plot(kind='bar', ax=axs[2], color=['lightcoral', 'lightskyblue'])
axs[2].set_title('Special Characters')
axs[2].set_xlabel('Strength')

plt.show()

import matplotlib.pyplot as plt # Import the necessary module for plotting.
import pandas as pd # Import pandas to work with Dataframes
import seaborn as sns # Import the seaborn library and alias it as 'sns'

#Load data again
df= pd.read_csv('passwords_dataset.csv')
# Stacked Bar Chart for Strength vs Special Characters
fig, ax = plt.subplots(figsize=(10, 6))
df_special_strength = df.groupby(['Strength', 'Has Special Character']).size().unstack().fillna(0)
df_special_strength.div(df_special_strength.sum(1), axis=0).plot(kind='bar', stacked=True, ax=ax, color=['lightcoral', 'lightskyblue'])
ax.set_title('Proportion of Passwords with Special Characters by Strength')
ax.set_xlabel('Strength')
ax.set_ylabel('Proportion')
plt.show()

# Box Plot for Length vs Special Characters
plt.figure(figsize=(10, 6))
sns.boxplot(x='Has Special Character', y='Length', data=df, palette='Set2')
plt.title('Password Length Distribution by Presence of Special Characters')
plt.xlabel('Has Special Character')
plt.ylabel('Length')
plt.show()

# Frequency of Special Characters in Strong Passwords
import collections

# Filter strong passwords
strong_passwords = df[df['Strength'] == 'Strong']['Password']

# Count special characters in strong passwords
special_chars = ''.join([c for pw in strong_passwords for c in pw if not c.isalnum()])
special_char_counts = collections.Counter(special_chars)

# Plot histogram of special character frequencies
plt.figure(figsize=(12, 6))
plt.bar(special_char_counts.keys(), special_char_counts.values(), color='skyblue')
plt.title('Frequency of Special Characters in Strong Passwords')
plt.xlabel('Special Characters')
plt.ylabel('Frequency')
plt.show()

# Redefine the central tendency values
mean_length = df['Length'].mean()
median_length = df['Length'].median()
mode_length = df['Length'].mode()[0]

# Histogram to visualise the distribution of password lengths
plt.figure(figsize=(10, 6))
df['Length'].plot(kind='hist', bins=range(8, 18), color='lightgreen', edgecolor='black')
plt.axvline(mean_length, color='blue', linestyle='dashed', linewidth=2, label=f'Mean: {mean_length:.2f}')
plt.axvline(median_length, color='red', linestyle='dashed', linewidth=2, label=f'Median: {median_length:.2f}')
plt.axvline(mode_length, color='purple', linestyle='dashed', linewidth=2, label=f'Mode: {mode_length}')
plt.title('Histogram of Password Lengths')
plt.xlabel('Password Length')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plotting the measures of central tendency and dispersion

# Box plot to visualise the quartiles and the spread of the data
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Length'], palette='Set2')
plt.title('Box Plot of Password Lengths')
plt.xlabel('Password Length')
plt.show()

# Redefine the quartiles since they might not be in the current context
quartiles_length = df['Length'].quantile([0.25, 0.5, 0.75])

# Histogram to visualise the distribution of password lengths with quartiles
plt.figure(figsize=(10, 6))
df['Length'].plot(kind='hist', bins=range(8, 18), color='lightgreen', edgecolor='black')
plt.axvline(mean_length, color='blue', linestyle='dashed', linewidth=2, label=f'Mean: {mean_length:.2f}')
plt.axvline(median_length, color='red', linestyle='dashed', linewidth=2, label=f'Median: {median_length:.2f}')
plt.axvline(mode_length, color='purple', linestyle='dashed', linewidth=2, label=f'Mode: {mode_length}')
plt.axvline(quartiles_length[0.25], color='orange', linestyle='dashed', linewidth=2, label=f'25th Percentile: {quartiles_length[0.25]}')
plt.axvline(quartiles_length[0.5], color='green', linestyle='dashed', linewidth=2, label=f'50th Percentile: {quartiles_length[0.5]}')
plt.axvline(quartiles_length[0.75], color='brown', linestyle='dashed', linewidth=2, label=f'75th Percentile: {quartiles_length[0.75]}')
plt.title('Histogram of Password Lengths with Quartiles')
plt.xlabel('Password Length')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Calculate entropy for each password
import math

def calculate_entropy(password):
  """Calculates the Shannon entropy of a password."""
  char_counts = collections.Counter(password)
  total_chars = len(password)
  entropy = 0
  for count in char_counts.values():
    probability = count / total_chars
    entropy -= probability * math.log2(probability)
  return entropy

df['Entropy'] = df['Password'].apply(calculate_entropy) # Calculate and add the 'Entropy' column to the DataFrame


# Plotting the histogram of password entropy
plt.figure(figsize=(10, 6))
df['Entropy'].plot(kind='hist', bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Password Entropy')
plt.xlabel('Entropy')
plt.ylabel('Frequency')
plt.show()

# Box plot to compare entropy across different strength categories
plt.figure(figsize=(10, 6))
sns.boxplot(x='Strength', y='Entropy', data=df, palette='Set2')
plt.title('Password Entropy Distribution by Strength')
plt.xlabel('Strength')
plt.ylabel('Entropy')
plt.show()

# Scatter plot to visualise the relationship between entropy and strength
plt.figure(figsize=(10, 6))
colors = {'Weak': 'red', 'Medium': 'orange', 'Strong': 'green'}
plt.scatter(df['Entropy'], df['Strength'].map(colors))
plt.title('Scatter Plot of Password Entropy and Strength')
plt.xlabel('Entropy')
plt.ylabel('Strength')
plt.show()

from scipy.stats import pearsonr

# Calculate the correlation coefficient between length and entropy
correlation, p_value = pearsonr(df['Length'], df['Entropy'])

# Scatter plot to visualise the correlation
plt.figure(figsize=(10, 6))
plt.scatter(df['Length'], df['Entropy'], alpha=0.5, color='purple')
plt.title(f'Scatter Plot of Password Length and Entropy\nCorrelation: {correlation:.2f}, p-value: {p_value:.2e}')
plt.xlabel('Password Length')
plt.ylabel('Entropy')
plt.show()

from scipy.stats import f_oneway

# Conduct ANOVA
anova_result = f_oneway(df[df['Strength'] == 'Weak']['Entropy'],
                        df[df['Strength'] == 'Medium']['Entropy'],
                        df[df['Strength'] == 'Strong']['Entropy'])

# Box plot to visualise the entropy distribution by strength
plt.figure(figsize=(10, 6))
sns.boxplot(x='Strength', y='Entropy', data=df, palette='Set2')
plt.title(f'Password Entropy Distribution by Strength\nANOVA p-value: {anova_result.pvalue:.2e}')
plt.xlabel('Strength')
plt.ylabel('Entropy')
plt.show()

from scipy.stats import chi2_contingency

# Contingency tables for chi-square tests
contingency_table_lowercase = pd.crosstab(df['Strength'], df['Has Lowercase'])
contingency_table_uppercase = pd.crosstab(df['Strength'], df['Has Uppercase'])
contingency_table_special = pd.crosstab(df['Strength'], df['Has Special Character'])

# Perform chi-square tests
chi2_lowercase, p_lowercase, _, _ = chi2_contingency(contingency_table_lowercase)
chi2_uppercase, p_uppercase, _, _ = chi2_contingency(contingency_table_uppercase)
chi2_special, p_special, _, _ = chi2_contingency(contingency_table_special)

# Display the results
contingency_table_lowercase, chi2_lowercase, p_lowercase, contingency_table_uppercase, chi2_uppercase, p_uppercase, contingency_table_special, chi2_special, p_special


from scipy.stats import ttest_ind

# T-test between Weak and Medium
t_stat_weak_medium, p_value_weak_medium = ttest_ind(df[df['Strength'] == 'Weak']['Entropy'],
                                                    df[df['Strength'] == 'Medium']['Entropy'])

# T-test between Medium and Strong
t_stat_medium_strong, p_value_medium_strong = ttest_ind(df[df['Strength'] == 'Medium']['Entropy'],
                                                        df[df['Strength'] == 'Strong']['Entropy'])

# T-test between Weak and Strong
t_stat_weak_strong, p_value_weak_strong = ttest_ind(df[df['Strength'] == 'Weak']['Entropy'],
                                                    df[df['Strength'] == 'Strong']['Entropy'])

# Display the results
ttest_results = pd.DataFrame({
    "Comparison": ["Weak vs Medium", "Medium vs Strong", "Weak vs Strong"],
    "T-Statistic": [t_stat_weak_medium, t_stat_medium_strong, t_stat_weak_strong],
    "p-value": [p_value_weak_medium, p_value_medium_strong, p_value_weak_strong]
})
ttest_results

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
file_path = 'passwords_dataset.csv'
data = pd.read_csv(file_path)

# Function to check if a password contains special characters
data['Has Special Character'] = data['Password'].apply(lambda x: any(not c.isalnum() for c in x))

# Separate the data into two groups: passwords with special characters and without special characters
with_special = data[data['Has Special Character'] == True]['Length']
without_special = data[data['Has Special Character'] == False]['Length']

# Perform a two-sample t-test
t_statistic, p_value = stats.ttest_ind(with_special, without_special, equal_var=False)
print(f"T-statistic: {t_statistic}, P-value: {p_value}")

# Display the means of both groups
mean_with_special = with_special.mean()
mean_without_special = without_special.mean()

print(f"Mean Length of Passwords with Special Characters: {mean_with_special}")
print(f"Mean Length of Passwords without Special Characters: {mean_without_special}")

# Plotting the distributions
plt.figure(figsize=(10, 6))
sns.histplot(with_special, kde=True, color='blue', label='With Special Characters')
sns.histplot(without_special, kde=True, color='red', label='Without Special Characters')
plt.title('Distribution of Password Lengths with and without Special Characters')
plt.xlabel('Password Length')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# prompt:  define the df in the following code: import matplotlib.pyplot as plt
# # Scatter plot for Length vs Strength
# plt.figure(figsize=(10, 6))
# colors = {'Weak': 'red', 'Medium': 'orange', 'Strong': 'green'}
# strength_color = df_confirm['Strength'].map(colors)
# plt.scatter(df_confirm['Length'], df_confirm['Strength'].map(colors), alpha=0.5)
# plt.title('Scatter Plot of Password Length and Strength')
# p


df_confirm = data.copy()

import matplotlib.pyplot as plt
# Scatter plot for Length vs Strength
plt.figure(figsize=(10, 6))
colors = {'Weak': 'red', 'Medium': 'orange', 'Strong': 'green'}
strength_color = df_confirm['Strength'].map(colors)
plt.scatter(df_confirm['Length'], df_confirm['Strength'].map(colors), alpha=0.5)
plt.title('Scatter Plot of Password Length and Strength')
plt.xlabel('Length')
plt.ylabel('Strength')
plt.savefig('scatter_plot_length_vs_strength.png')
plt.show()

import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt


df_confirm = data.copy()

# Prepare the data for regression - using 'Strength' as a numerical proxy
#  map 'Strength' to numerical values if it's categorical
if df_confirm['Strength'].dtype == object:  # Check if 'Strength' is categorical
    strength_mapping = {'Weak': 1, 'Medium': 2, 'Strong': 3}  # Example mapping
    df_confirm['Strength'] = df_confirm['Strength'].map(strength_mapping)

X = df_confirm['Length']
y = df_confirm['Strength']  # Using 'Strength' as the dependent variable
X = sm.add_constant(X)  # Add a constant term for the intercept

# Fit the regression model
model = sm.OLS(y, X).fit()

# Get the regression line
df_confirm['Regression Line'] = model.predict(X)

# Scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(df_confirm['Length'], df_confirm['Strength'], alpha=0.5)
plt.plot(df_confirm['Length'], df_confirm['Regression Line'], color='red')
plt.title('Scatter Plot of Password Length vs Strength with Regression Line')
plt.xlabel('Length')
plt.ylabel('Strength')
plt.show()


