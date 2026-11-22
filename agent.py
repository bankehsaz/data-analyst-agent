
import json
import inspect
import math

import pandas as pd
import ollama

import tools


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

MODEL = "qwen2.5:0.5b"

MAX_AGENT_ITERATIONS = 5

client = ollama.Client(host="http://localhost:11434")


# --------------------------------------------------
# 2. JSON Serialization
# --------------------------------------------------

def make_json_safe(value):
    """
    Convert Python, Pandas and NumPy values
    into JSON-compatible values.
    """

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, (pd.Series,)):
        return make_json_safe(value.to_dict())

    if isinstance(value, (pd.DataFrame,)):
        return make_json_safe(value.to_dict(orient="records"))

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass

    if isinstance(value, float) and not math.isfinite(value):
        return None

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def to_json(value):
    """Convert a value into a JSON string."""

    return json.dumps(
        make_json_safe(value),
        ensure_ascii=False,
        allow_nan=False
    )


# --------------------------------------------------
# 3. Tool Definitions
# --------------------------------------------------

def create_tool(name, description, properties=None, required=None):
    """Create a JSON Schema tool definition."""

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or []
            }
        }
    }


TOOLS = [

    # Dataset Overview

    create_tool(
        "get_data_info",
        "Get the dataset dimensions, column names, data types and missing values."
    ),

    create_tool(
        "get_data_preview",
        "Get the first rows of the dataset.",
        {
            "n": {
                "type": "integer",
                "description": "Number of rows to return."
            }
        }
    ),

    create_tool(
        "get_numeric_columns",
        "Get the names of all numeric columns."
    ),

    create_tool(
        "get_categorical_columns",
        "Get the names of all categorical columns."
    ),

    # Descriptive Statistics

    create_tool(
        "get_descriptive_stats",
        "Calculate descriptive statistics for all dataset columns."
    ),

    create_tool(
        "get_column_summary",
        "Calculate descriptive statistics for a specific column.",
        {
            "column": {
                "type": "string",
                "description": "Column name."
            }
        },
        ["column"]
    ),

    create_tool(
        "get_value_counts",
        "Calculate the frequency of each value in a column.",
        {
            "column": {
                "type": "string",
                "description": "Column name."
            }
        },
        ["column"]
    ),

    create_tool(
        "get_unique_values",
        "Get the unique non-missing values in a column.",
        {
            "column": {
                "type": "string",
                "description": "Column name."
            }
        },
        ["column"]
    ),

    # Data Quality

    create_tool(
        "get_missing_values",
        "Count missing values in every column."
    ),

    create_tool(
        "get_duplicate_info",
        "Get the number of duplicate rows and total rows."
    ),

    create_tool(
        "get_unique_counts",
        "Count unique values in every column."
    ),

    # Correlation Analysis

    create_tool(
        "get_correlation",
        "Calculate the correlation matrix for numeric columns."
    ),

    # Data Visualization

    create_tool(
        "create_histogram",
        "Create a histogram to show the distribution of a numeric column.",
        {
            "column": {
                "type": "string",
                "description": "Numeric column name."
            }
        },
        ["column"]
    ),

    create_tool(
        "create_boxplot",
        "Create a boxplot for a numeric column to inspect its distribution and outliers.",
        {
            "column": {
                "type": "string",
                "description": "Numeric column name."
            }
        },
        ["column"]
    ),

    create_tool(
        "create_scatter_plot",
        "Create a scatter plot to examine the relationship between two columns.",
        {
            "x_column": {
                "type": "string",
                "description": "Column for the X axis."
            },
            "y_column": {
                "type": "string",
                "description": "Column for the Y axis."
            }
        },
        ["x_column", "y_column"]
    ),

    create_tool(
        "create_bar_chart",
        "Create a bar chart showing the most frequent values in a column.",
        {
            "column": {
                "type": "string",
                "description": "Column name."
            },
            "top_n": {
                "type": "integer",
                "description": "Maximum number of categories to display."
            }
        },
        ["column"]
    ),

    create_tool(
        "create_line_chart",
        "Create a line chart showing the relationship between two columns.",
        {
            "x_column": {
                "type": "string",
                "description": "Column for the X axis."
            },
            "y_column": {
                "type": "string",
                "description": "Column for the Y axis."
            }
        },
        ["x_column", "y_column"]
    ),

    create_tool(
        "create_correlation_heatmap",
        "Create a heatmap to visualize correlations between numeric columns."
    )
]


# --------------------------------------------------
# 4. Available Python Functions
# --------------------------------------------------

AVAILABLE_FUNCTIONS = {

    # Dataset Overview
    "get_data_info": tools.get_data_info,
    "get_data_preview": tools.get_data_preview,
    "get_numeric_columns": tools.get_numeric_columns,
    "get_categorical_columns": tools.get_categorical_columns,

    # Descriptive Statistics
    "get_descriptive_stats": tools.get_descriptive_stats,
    "get_column_summary": tools.get_column_summary,
    "get_value_counts": tools.get_value_counts,
    "get_unique_values": tools.get_unique_values,

    # Data Quality
    "get_missing_values": tools.get_missing_values,
    "get_duplicate_info": tools.get_duplicate_info,
    "get_unique_counts": tools.get_unique_counts,

    # Correlation Analysis
    "get_correlation": tools.get_correlation,

    # Data Visualization
    "create_histogram": tools.create_histogram,
    "create_boxplot": tools.create_boxplot,
    "create_scatter_plot": tools.create_scatter_plot,
    "create_bar_chart": tools.create_bar_chart,
    "create_line_chart": tools.create_line_chart,
    "create_correlation_heatmap": tools.create_correlation_heatmap
}


# --------------------------------------------------
# 5. Tool Execution
# --------------------------------------------------

CHART_FUNCTIONS = {
    "create_histogram",
    "create_boxplot",
    "create_scatter_plot",
    "create_bar_chart",
    "create_line_chart",
    "create_correlation_heatmap"
}


def execute_tool(tool_name, arguments, df, charts):
    """
    Validate and execute a requested Python tool.
    """

    if tool_name not in AVAILABLE_FUNCTIONS:
        return f"Error: Unknown tool '{tool_name}'."

    function = AVAILABLE_FUNCTIONS[tool_name]

    if not isinstance(arguments, dict):
        return "Error: Tool arguments must be a dictionary."

    try:
        # Validate function arguments
        signature = inspect.signature(function)

        allowed_parameters = set(signature.parameters.keys())

        invalid_arguments = set(arguments.keys()) - allowed_parameters

        if invalid_arguments:
            return (
                "Error: Invalid arguments: "
                + ", ".join(invalid_arguments)
            )

        # Validate column names
        column_arguments = [
            "column",
            "x_column",
            "y_column"
        ]

        for argument_name in column_arguments:
            if argument_name in arguments:
                column_name = arguments[argument_name]

                if column_name not in df.columns:
                    return (
                        f"Error: Column '{column_name}' "
                        "does not exist in the dataset."
                    )

        # Execute visualization tools
        if tool_name in CHART_FUNCTIONS:

            result = function(df, **arguments)

            charts.append((result, tool_name))

            return (
                f"Chart '{tool_name}' was created successfully. "
                "The chart is ready to be displayed."
            )

        # Execute analytical tools
        result = function(df, **arguments)

        return to_json(result)

    except Exception as error:
        return f"Error while executing '{tool_name}': {str(error)}"


# --------------------------------------------------
# 6. Agent System Prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a Data Analyst Agent.

Your task is to analyze datasets using the available Python tools.

IMPORTANT RULES:

1. Always use tools to obtain actual data and calculate statistics.
   Never guess or invent results.

2. The dataset is already loaded.
   You do not need to load the dataset again.

3. The Python tools automatically receive the dataset.
   Never pass the dataset as an argument to a tool.

4. Use only the exact column names provided by the dataset.

5. Choose the correct tool for each task.

TOOL SELECTION GUIDELINES:

- For general dataset information:
  Use get_data_info.

- For dataset preview:
  Use get_data_preview.

- For numerical columns:
  Use get_numeric_columns.
  This tool requires no arguments.

- For categorical columns:
  Use get_categorical_columns.
  This tool requires no arguments.

- For descriptive statistics of all columns:
  Use get_descriptive_stats.
  This tool requires no arguments.

- For statistics of a specific column:
  Use get_column_summary.
  Pass the column name using the "column" argument.

- For frequency counts of a column:
  Use get_value_counts.
  Pass the column name using the "column" argument.

- For unique values of a column:
  Use get_unique_values.
  Pass the column name using the "column" argument.

- For missing values:
  Use get_missing_values.
  This tool requires no arguments.

- For duplicate rows:
  Use get_duplicate_info.
  This tool requires no arguments.

- For correlation analysis:
  Use get_correlation.
  This tool requires no arguments.

VISUALIZATION GUIDELINES:

- For numerical data distribution:
  Use create_histogram.

- For identifying outliers:
  Use create_boxplot.

- For relationships between two numerical columns:
  Use create_scatter_plot.

- For comparing category frequencies:
  Use create_bar_chart.

- For trends over an ordered variable:
  Use create_line_chart.

- For correlation visualization:
  Use create_correlation_heatmap.

When using visualization tools, always use valid column names
and provide only the arguments required by the tool.

If the user requests both statistical analysis and a chart,
perform both tasks.

Explain the results clearly and accurately.

If a tool returns an error, do not pretend the operation succeeded.

Answer in the same language as the user's question.
"""


# --------------------------------------------------
# 7. Main Agent Function
# --------------------------------------------------

def get_agent_response(user_prompt, df):
    """
    Process a user request using Ollama and Python tools.

    Returns:
        response_text: Final response from the model.
        charts: List of generated Matplotlib figures.
    """

    charts = []

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    try:

        # Tool Calling Loop
        for _ in range(MAX_AGENT_ITERATIONS):

            response = client.chat(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                options={
                    "temperature": 0.2
                }
            )

            assistant_message = response.message

            # Save assistant response
            messages.append(assistant_message)

            tool_calls = assistant_message.tool_calls or []

            # No more tool calls
            if not tool_calls:
                return assistant_message.content or (
                    "The model did not generate a response."
                ), charts

            # Execute requested tools
            for tool_call in tool_calls:

                tool_name = tool_call.function.name
                arguments = tool_call.function.arguments

                result = execute_tool(
                    tool_name=tool_name,
                    arguments=arguments,
                    df=df,
                    charts=charts
                )

                # Send tool result back to the model
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": str(result)
                    }
                )

        # Maximum tool calls reached
        messages.append(
            {
                "role": "user",
                "content": (
                    "The tool execution limit has been reached. "
                    "Please provide your final answer using "
                    "the results already available."
                )
            }
        )

        final_response = client.chat(
            model=MODEL,
            messages=messages,
            options={
                "temperature": 0.2
            }
        )

        return final_response.message.content, charts

    except Exception as error:

        return (
            f"An error occurred while communicating with Ollama: {str(error)}",
            charts
        )
