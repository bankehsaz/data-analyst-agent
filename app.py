
import streamlit as st
import matplotlib.pyplot as plt

from tools import load_data, get_data_info
from agent import get_agent_response


# ==========================================
# 1. Page Configuration
# ==========================================

st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# 2. Application Title
# ==========================================

st.title("📊 Data Analyst Agent")

st.write(
    "Upload a CSV dataset and ask questions "
    "about your data using natural language."
)


# ==========================================
# 3. File Upload
# ==========================================

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)


# ==========================================
# 4. Load and Preview Dataset
# ==========================================

if uploaded_file is not None:

    try:
        uploaded_file.seek(0)
        df = load_data(uploaded_file)

        st.success("Dataset loaded successfully!")

        st.subheader("Dataset Preview")

        st.dataframe(df.head(), width="stretch")

        # Dataset Overview
        info = get_data_info(df)

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", info["rows"])
        col2.metric("Columns", info["columns"])
        col3.metric(
            "Missing Values",
            sum(info["missing_values"].values())
        )

        with st.expander("Dataset Information"):

            st.write("Column Names:")
            st.write(info["column_names"])

            st.write("Data Types:")
            st.write(info["data_types"])

            st.write("Missing Values:")
            st.write(info["missing_values"])


        # ==========================================
        # 5. User Question
        # ==========================================

        st.divider()

        st.subheader("Ask Your Data Analyst Agent")

        user_prompt = st.text_area(
            "Enter your question:",
            placeholder=(
                "Example: What is the average value "
                "of the price column?"
            )
        )


        # ==========================================
        # 6. Generate Agent Response
        # ==========================================

        if st.button("Analyze Data"):

            if not user_prompt.strip():

                st.warning("Please enter a question.")

            else:

                with st.spinner("Analyzing your data..."):

                    response_text, charts = get_agent_response(
                        user_prompt,
                        df
                    )

                st.subheader("Agent Response")

                st.markdown(response_text)


                # ==========================================
                # 7. Display Generated Charts
                # ==========================================

                if charts:

                    st.subheader("Generated Charts")

                    for fig, chart_name in charts:

                        st.write(
                            f"Chart: {chart_name.replace('_', ' ').title()}"
                        )

                        st.pyplot(fig, width=600)

                        plt.close(fig)


    except Exception as error:

        st.error(f"An error occurred: {str(error)}")


else:

    st.info("Please upload a CSV file to get started.")
