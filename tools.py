
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. Data Loading
# ==========================================

def load_data(file):
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(file)


# ==========================================
# 2. Dataset Overview
# ==========================================

def get_data_info(df):
    """Get general information about the dataset."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict()
    }


def get_data_preview(df, n=5):
    """Get the first n rows of the dataset."""
    return df.head(n).to_dict(orient="records")


def get_numeric_columns(df):
    """Get numeric column names."""
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(df):
    return df.select_dtypes(
        include=["object", "category", "string", "str"]
    ).columns.tolist()


# ==========================================
# 3. Descriptive Statistics
# ==========================================

def get_descriptive_stats(df):
    """Calculate descriptive statistics."""
    return df.describe(include="all").to_dict()


def get_column_summary(df, column):
    """Get statistical summary of a specific column."""
    return df[column].describe().to_dict()


def get_value_counts(df, column):
    """Get frequency of unique values in a column."""
    return df[column].value_counts(dropna=False).to_dict()


def get_unique_values(df, column):
    """Get unique values of a column."""
    return df[column].dropna().unique().tolist()


# ==========================================
# 4. Data Quality Analysis
# ==========================================

def get_missing_values(df):
    """Count missing values in each column."""
    return df.isnull().sum().to_dict()


def get_duplicate_info(df):
    """Get duplicate row information."""
    return {
        "duplicate_rows": int(df.duplicated().sum()),
        "total_rows": len(df)
    }


def get_unique_counts(df):
    """Count unique values in each column."""
    return df.nunique().to_dict()


# ==========================================
# 5. Correlation Analysis
# ==========================================

def get_correlation(df):
    """Calculate correlation between numeric columns."""
    return df.corr(numeric_only=True).to_dict()


# ==========================================
# 6. Data Visualization
# ==========================================

def create_histogram(df, column):
    """Create a histogram for a numeric column."""
    fig, ax = plt.subplots()

    df[column].dropna().hist(ax=ax, bins=20)

    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    fig.tight_layout()
    return fig


def create_boxplot(df, column):
    """Create a boxplot for a numeric column."""
    fig, ax = plt.subplots()

    df.boxplot(column=column, ax=ax)

    ax.set_title(f"Boxplot of {column}")
    ax.set_ylabel(column)

    fig.tight_layout()
    return fig


def create_scatter_plot(df, x_column, y_column):
    """Create a scatter plot between two numeric columns."""

    plot_data = df[[x_column, y_column]].dropna()

    fig, ax = plt.subplots()

    ax.scatter(
        plot_data[x_column],
        plot_data[y_column],
        alpha=0.7
    )

    ax.set_title(f"{y_column} vs {x_column}")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

    fig.tight_layout()
    return fig


def create_bar_chart(df, column, top_n=10):
    """Create a bar chart for categorical data."""
    fig, ax = plt.subplots()

    df[column].value_counts().head(top_n).plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(f"Top {top_n} Values in {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    fig.tight_layout()
    return fig


def create_line_chart(df, x_column, y_column):
    """Create a line chart between two columns."""
    fig, ax = plt.subplots()

    ax.plot(df[x_column], df[y_column], marker="o")

    ax.set_title(f"{y_column} over {x_column}")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

    fig.tight_layout()
    return fig


def create_correlation_heatmap(df):
    """Create a correlation heatmap for numeric columns."""
    corr = df.corr(numeric_only=True)

    if corr.empty:
        raise ValueError("No numeric columns available for correlation.")

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))

    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    fig.colorbar(image, ax=ax)

    ax.set_title("Correlation Heatmap")

    fig.tight_layout()
    return fig
