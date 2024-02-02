import ast
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import os
import pandas as pd
import re


def read_csv_files_to_dict(directory_path):
    """
    Recursively reads all CSV files in the given directory and its subdirectories,
    storing them in a dictionary.

    Args:
    directory_path (str): The path to the root directory containing CSV files.

    Returns:
    dict: A dictionary where keys are file names (including subdirectory path) and values are DataFrames of the CSV files.
    """
    csv_files_dict = {}

    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory does not exist: {directory_path}")
        return csv_files_dict

    # Walk through the directory and its subdirectories
    for dirpath, _, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.csv'):

                if filename in csv_files_dict.keys():
                    raise ValueError(f"You got two files called {filename} here mate")

                # read in the file and store in the output dict
                file_path = os.path.join(dirpath, filename)
                df = pd.read_csv(file_path)
                csv_files_dict[filename] = df

    return csv_files_dict


def split_text(df, text_col, max_word_length):
    """
    Splits texts in a DataFrame that are longer than a specified word length into smaller chunks,
    attempting to split at sentence boundaries or other punctuations.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing texts.
    text_col : str
        The column name containing the texts.
    max_word_length : int
        The maximum word length allowed for a single text.

    Returns
    -------
    pd.DataFrame
        A DataFrame with excessively long texts split into smaller chunks.
    """
    def split_text(text):
        words = text.split()
        start = 0
        while start < len(words):
            end = min(start + max_word_length, len(words))
            chunk = ' '.join(words[start:end])
            # Try to find a sentence end within the chunk
            boundary = max(chunk.rfind('.'), chunk.rfind('\n'), chunk.rfind(','),
                           chunk.rfind(';'), chunk.rfind(':'))
            if boundary != -1:
                end = start + len(chunk[:boundary].split())
            yield ' '.join(words[start:end])
            start = end

    new_rows = []
    for _, row in df.iterrows():
        text_chunks = list(split_text(row[text_col]))
        for chunk in text_chunks:
            new_row = row.copy()
            new_row[text_col] = chunk
            new_rows.append(new_row)

    return pd.DataFrame(new_rows).reset_index(drop=True)


def expand_labels(df):
    """
    Expands a DataFrame by converting a column of string-formatted dictionaries 
    into separate columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing a column of string-formatted dictionaries.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the string-formatted dictionary column 
        expanded into separate columns.
    """
    df['labels'] = df['labels'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else {}
    ) # Convert string-formatted dictionary to dictionary
    expanded_data = pd.json_normalize(df['labels']) # Normalize dictionary entries to a separate DataFrame
    expanded_data.index = df.index
    result_df = pd.concat([df, expanded_data], axis=1)
    result_df = result_df.drop(columns=['labels'])
    return result_df


def unexpand_labels(df, label_cols):
    """
    Reverts label columns back to a single 'labels' column in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing label columns.
    label_cols : list of str
        A list of column names to be included in the 'labels' dictionary.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with a single 'labels' column containing the 
        reconstructed dictionaries.
    """

    # Create a new DataFrame to store the reconstructed dictionaries
    label_dicts = pd.DataFrame(index=df.index)

    # Iterate over the label columns and construct the dictionaries
    for col in label_cols:
        label_dicts[col] = df[col]

    # Convert the rows of label_dicts DataFrame to dictionary
    df['labels'] = label_dicts.apply(lambda x: x.dropna().to_dict(), axis=1)

    # Convert dictionaries to their string representation
    df['labels'] = df['labels'].apply(lambda x: str(x) if x else '{}')

    # Drop the label columns
    df = df.drop(columns=label_cols)

    return df


def create_data_viewer(df, page_size=10):
    """
    Creates an interactive data viewer for a DataFrame using IPython widgets. 
    This viewer allows browsing through the DataFrame entries page by page and 
    provides functionality to delete rows or swap the values of a specified column. 
    It is designed to assist in the manual review and editing of data entries, 
    particularly useful in data cleaning and annotation tasks.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be viewed and edited. It must contain the columns 
        'text', 'label', 'prob', and 'loss'. 
    page_size : int, optional
        The number of rows to display on each page of the viewer. 
        The default value is 10.

    Returns
    -------
    None
        This function does not return any value. It directly updates the `df` 
        given as input by deleting selected rows or swapping label values based 
        on user interaction within the generated widget interface.
    """
    current_start = 0  # Tracks the current starting index of rows being displayed.

    # Function to create widgets for displaying each row's data.
    def create_row_widgets(row, delete_checkboxes, label_checkboxes):
        # Import necessary modules inside the function to avoid global dependencies.
        from IPython.display import display, clear_output
        import ipywidgets as widgets

        # Create a text area widget to display the text of the row.
        text_w = widgets.Textarea(value=row['text'], disabled=True, layout=widgets.Layout(width='40%', height='100px'))
        # Create label widgets to display the label, probability, and loss values.
        label_w = widgets.Label(value=f"Label: {row['label']}", layout=widgets.Layout(width='10%'))
        prob_w = widgets.Label(value=f"Prob: {row['prob']:.2f}", layout=widgets.Layout(width='10%'))
        loss_w = widgets.Label(value=f"Loss: {row['loss']:.2f}", layout=widgets.Layout(width='10%'))

        # Create checkboxes for deletion and label swapping actions.
        delete_check = widgets.Checkbox(value=False, description='Delete')
        swap_label_check = widgets.Checkbox(value=False, description='Swap Label')

        # Store references to these checkboxes for later processing.
        delete_checkboxes.append((row.name, delete_check))
        label_checkboxes.append((row.name, swap_label_check))

        # Return a horizontal box containing all the widgets for a single row.
        return widgets.HBox([text_w, label_w, prob_w, loss_w, swap_label_check, delete_check],
                            layout=widgets.Layout(display='flex', flex_flow='row wrap'))

    # Function to display data for the current page and setup navigation.
    def display_data(start=0):
        nonlocal current_start  # Reference to track the current page start index.
        clear_output(wait=True)  # Clear the previous output to display the current page cleanly.
        end = start + page_size  # Calculate the ending index for the current page.
        # Calculate indices to display, wrapping around if necessary.
        rows_to_display = (list(range(start, len(df))) + list(range(0, end % len(df))))[:page_size]
        current_df = df.iloc[rows_to_display]  # Extract rows for the current page.
        delete_checkboxes = []  # List to store delete checkboxes for current page.
        label_checkboxes = []  # List to store label swap checkboxes for current page.

        # Create and display widgets for each row in the current page.
        for _, row in current_df.iterrows():
            row_widget = create_row_widgets(row, delete_checkboxes, label_checkboxes)
            display(row_widget)

        # Display navigation buttons (previous, save, next) for the page.
        navigation_buttons = create_navigation_buttons(delete_checkboxes, label_checkboxes)
        display(navigation_buttons)

    # Function to create and handle navigation and action buttons.
    def create_navigation_buttons(delete_checkboxes, label_checkboxes):
        import ipywidgets as widgets
        from IPython.display import display

        # Create buttons for saving changes, and navigating to the next and previous pages.
        save_button = widgets.Button(description="Save Changes")
        next_button = widgets.Button(description="Next Page")
        prev_button = widgets.Button(description="Previous Page")

        # Define button click actions: save changes, move to next, and move to previous page.
        def on_save(b):
            nonlocal current_start
            # Process deletions and label swaps based on checkbox values.
            for index, checkbox in delete_checkboxes + label_checkboxes:
                if checkbox.value:  # If checkbox is checked...
                    if checkbox.description == 'Delete':
                        df.drop(index, inplace=True)  # Delete the row.
                    elif checkbox.description == 'Swap Label':
                        # Swap the label value (1 to 0, or vice versa).
                        df.at[index, 'label'] = int(not df.at[index, 'label'])

            # After saving, display the next page of data.
            current_start = (current_start + page_size) % len(df)
            display_data(current_start)

        def on_next(b):
            nonlocal current_start
            # Move to the next page, wrapping around if necessary.
            current_start = (current_start + page_size) % len(df)
            display_data(current_start)

        def on_prev(b):
            nonlocal current_start
            # Move to the previous page, wrapping to the end if necessary.
            current_start = (current_start - page_size) % len(df)
            display_data(current_start)

        # Attach the defined actions to the buttons.
        save_button.on_click(on_save)
        next_button.on_click(on_next)
        prev_button.on_click(on_prev)

        # Return a horizontal box containing the navigation buttons.
        return widgets.HBox([prev_button, save_button, next_button])

    # Initial call to display the first page of data.
    display_data(current_start)
