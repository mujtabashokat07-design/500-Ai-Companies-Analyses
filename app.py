"""
Enterprise AI Adoption Dashboard
Author: Data Analytics & Engineering Team
Description: A production-ready Streamlit application built to analyze AI adoption, 
             financial ROI percentages, and organizational maturity scores across 
             global Fortune 500 cohorts (2020-2025).
"""

import logging
from typing import Final
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Configure enterprise logging mechanism
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global Configuration and Constants
DATASET_PATH: Final[str] = "ai-adoption-fortune500-synthetic-dataset-2020-2025.csv"
PLOT_STYLE: Final[str] = (
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "ggplot"
)


@st.cache_data(show_spinner="Ingesting and preprocessing enterprise records...")
def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """Loads target CSV dataset and executes structural transformation rules.

    Args:
        file_path (str): File system path pointing to source data asset.

    Returns:
        pd.DataFrame: Cleaned and structured pandas DataFrame ready for analytics.
    """
    try:
        logger.info(f"Initiating data ingestion pipeline for: {file_path}")
        df = pd.read_csv(file_path)

        # Handle implicit or explicit missing categorical labels structurally
        if "Use_Case" in df.columns:
            df["Use_Case"] = df["Use_Case"].fillna("No Use Case")

        logger.info("Data pipeline executed successfully. Records cached.")
        return df

    except FileNotFoundError:
        logger.critical(f"Data pipeline failed. File missing at: {file_path}")
        st.error(
            f"Critical Storage Error: The source target '{file_path}' could not be resolved."
        )
        st.stop()
    except Exception as error:
        logger.exception(f"Unexpected execution exception: {str(error)}")
        st.error(
            "Application Engine Exception: Failed to clean or parse internal records."
        )
        st.stop()


def render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Renders highly interactive, dynamic filtering controls within the application sidebar.

    Args:
        df (pd.DataFrame): Unfiltered original base DataFrame.

    Returns:
        pd.DataFrame: Sliced and structured dataset matching multi-select logic arrays.
    """
    st.sidebar.header("🎯 Target Selection Controls")

    # Safe extraction of unique dimensional variables for filtering arrays
    available_years = (
        sorted(df["Year"].unique()) if "Year" in df.columns else []
    )
    available_countries = (
        sorted(df["Country"].unique()) if "Country" in df.columns else []
    )
    available_industries = (
        sorted(df["Industry"].unique()) if "Industry" in df.columns else []
    )

    # Multi-select control UI elements
    selected_years = st.sidebar.multiselect(
        label="Reporting Operational Year",
        options=available_years,
        default=available_years,
    )
    selected_countries = st.sidebar.multiselect(
        label="Geographic Region / Market",
        options=available_countries,
        default=available_countries,
    )
    selected_industries = st.sidebar.multiselect(
        label="Target Industry Sector Vertical",
        options=available_industries,
        default=available_industries,
    )

    # Boolean logical execution of slicing array matrix
    conditioned_slice = df[
        (df["Year"].isin(selected_years))
        & (df["Country"].isin(selected_countries))
        & (df["Industry"].isin(selected_industries))
    ]

    return conditioned_slice


def render_executive_metrics(df: pd.DataFrame) -> None:
    """Computes and updates core enterprise performance metric modules on UI grid.

    Args:
        df (pd.DataFrame): Currently active filtered data subset.
    """
    st.subheader("📊 Executive Target Telemetry")
    col1, col2, col3, col4 = st.columns(4)

    # Statistical computation variables
    total_firms = len(df)
    mean_roi = df["AI_ROI_Percent"].mean() if "AI_ROI_Percent" in df.columns else 0.0
    mean_maturity = (
        df["AI_Maturity_Score"].mean() if "AI_Maturity_Score" in df.columns else 0.0
    )
    active_adoption = (
        df[df["Uses_AI"] == "Yes"].shape[0] if "Uses_AI" in df.columns else 0
    )

    with col1:
        st.metric("Total Corporate Cohorts", f"{total_firms:,}")
    with col2:
        st.metric("Mean System ROI Index", f"{mean_roi:.2f}%")
    with col3:
        st.metric("Mean System Maturity", f"{mean_maturity:.2f} / 5.0")
    with col4:
        st.metric("Active System Scale", f"{active_adoption:,}")


def render_distribution_plots(df: pd.DataFrame) -> None:
    """Generates precise high-resolution frequency density visualization profiles.

    Args:
        df (pd.DataFrame): Cleaned and filtered workspace dataset.
    """
    st.subheader("📈 Statistical Probability & Density Profiles")

    plt.style.use(PLOT_STYLE)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Metric Subplot 1: Return on Investment Profile
    sns.histplot(
        data=df,
        x="AI_ROI_Percent",
        bins=25,
        kde=True,
        ax=axes[0],
        color="#1f77b4",
        edgecolor="#ffffff",
        alpha=0.85,
    )
    axes[0].set_title(
        "Density Profile: Financial Metrics (ROI %)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    axes[0].set_xlabel("Return on Investment Percentage")
    axes[0].set_ylabel("Frequency Distribution Density")

    # Metric Subplot 2: Organizational Competence Profile
    sns.histplot(
        data=df,
        x="AI_Maturity_Score",
        bins=20,
        kde=True,
        ax=axes[1],
        color="#ff7f0e",
        edgecolor="#ffffff",
        alpha=0.85,
    )
    axes[1].set_title(
        "Density Profile: Organizational Maturity Scale",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    axes[1].set_xlabel("Maturity Evaluation Score")
    axes[1].set_ylabel("Frequency Distribution Density")

    # Clean axes parameters and render object to stream container
    sns.despine(top=True, right=True)
    plt.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


def render_categorical_breakdowns(df: pd.DataFrame) -> None:
    """Constructs categorical distribution analysis charts across market dimensions.

    Args:
        df (pd.DataFrame): Active parsed system data.
    """
    st.subheader("🏢 Market Demographics & Sector Volatility")
    left_grid_col, right_grid_col = st.columns(2)

    plt.style.use(PLOT_STYLE)

    with left_grid_col:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(
            data=df,
            x="Company_Type",
            hue="Uses_AI",
            ax=ax,
            palette="bright",
            edgecolor="#000000",
            alpha=0.8,
        )
        ax.set_title(
            "Deployment Ingestion Status by Structural Entity Type",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Corporate Class Category")
        ax.set_ylabel("Operational Volumes")
        sns.despine(top=True, right=True)
        st.pyplot(fig)
        plt.close(fig)

    with right_grid_col:
        fig, ax = plt.subplots(figsize=(8, 5))
        active_adopters = df[df["Uses_AI"] == "Yes"]

        if not active_adopters.empty:
            top_sectors = active_adopters["Industry"].value_counts().head(10)
            sns.barplot(
                x=top_sectors.values,
                y=top_sectors.index,
                ax=ax,
                palette="dark:salmon_r",
                edgecolor="#000000",
                alpha=0.85,
            )
            ax.set_title(
                "Top 10 Vertical Segments Maximizing Strategy Deployments",
                fontsize=11,
                fontweight="bold",
                pad=10,
            )
            ax.set_xlabel("Total Active System Footprint")
            ax.set_ylabel("Industry Sector Classification")
            sns.despine(top=True, right=True)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(
                "No confirmed active adoption profiles exist inside current filtering parameter parameters."
            )


def main() -> None:
    """Core control module orchestrating global execution steps and application framework."""
    st.set_page_config(
        page_title="Corporate AI Analytics Hub", page_icon="🤖", layout="wide"
    )

    # Core Application Header Information
    st.title("🤖 Enterprise AI Adoption Intelligence Grid")
    st.markdown(
        "This diagnostic engine evaluates the structural transition metrics, financial scalability, "
        "and technological capabilities of Fortune 500 corporate entities across critical validation years."
    )
    st.divider()

    # Data ingestion layer execution
    raw_dataframe = load_and_preprocess_data(DATASET_PATH)

    # UI interaction filtering extraction
    filtered_dataframe = render_sidebar_filters(raw_dataframe)

    # Safe validation check for empty analytical slices
    if filtered_dataframe.empty:
        st.warning(
            "⚠️ Filter constraints resulted in zero active matching matrices. Expand parameters to process data."
        )
        return

    # Sequentially render dashboards components
    render_executive_metrics(filtered_dataframe)
    st.divider()

    # Operational Data Table Output Previewer
    st.subheader("📋 Active Telemetry Warehouse Ledger")
    st.dataframe(filtered_dataframe, use_container_width=True, hide_index=True)
    st.divider()

    # Graphics Visualization Layers
    render_distribution_plots(filtered_dataframe)
    st.divider()
    render_categorical_breakdowns(filtered_dataframe)


if __name__ == "__main__":
    main()