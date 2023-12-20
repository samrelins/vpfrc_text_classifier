import csv
from datetime import datetime
from IPython.display import HTML
import openai
import os
import time


def save_prompt_responses(
    prompt: str, responses: list, labels: dict, model_name: str, save_path: str,
    model_params: dict
):
    """
    Saves examples with a given label to a CSV file.

    Parameters
    ----------
    prompt : str
        The prompt text used to generate the responses.
    responses : list
        A list of response texts generated by the model.
    labels : dict
        A dictionary containing label information for each response.
    model_name : str
        The name of the model used to generate the responses.
    save_path : str
        The file path (including filename) where the CSV file will be saved.
    model_params : dict
        A dictionary containing the inference parameters for the model.

    Returns
    -------
    None

    Notes
    -----
    The function will append data to the CSV file if it already exists,
    otherwise, it will create a new CSV file with appropriate headers.
    """

    # Determine the file path
    file_path = save_path if save_path.endswith('.csv') else f"{save_path}.csv"

    # Check if the file exists to determine if headers are needed
    file_exists = os.path.exists(file_path)

    # Open the CSV file in append mode
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write headers if the file did not exist
        if not file_exists:
            headers = ['timestamp', 'model', 'model_params', 'prompt', 'response', 'labels']
            writer.writerow(headers)

        # Write data
        for response in responses:
            row = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                model_name,
                str(model_params),  # Convert dict to string for CSV
                prompt,
                response,
                str(labels)  # Convert dict to string for CSV
            ]
            writer.writerow(row)


def generate_prompt(examples_df, scenarios_df, k=1, additional_instruction=""):
    """
    Generates a prompt for creating example notes of calls to emergency services.

    Parameters
    ----------
    examples_df : DataFrame
        DataFrame containing example scenarios and logs.
    scenarios_df : DataFrame
        DataFrame containing different incident types.
    k : int, optional
        The number of examples to be included in the prompt. Defaults to 1.
    additional_instruction : str, optional
        Additional instruction or context to be appended to the prompt. Defaults to an empty string.

    Returns
    -------
    tuple
        A tuple containing:
            prompt (str): A string that forms the complete prompt, including examples and instructions.
            scenario (str): The scenario selected for the report.
    """

    if examples_df.empty or scenarios_df.empty:
            raise ValueError("Input DataFrames cannot be empty")

        # Ensuring unique scenarios are selected
    unique_scenarios = examples_df['scenario'].unique()
    if len(unique_scenarios) < k:
        raise ValueError("Not enough unique scenarios to select from")

    selected_scenarios = np.random.choice(unique_scenarios, k, replace=False)
    scenarios_reports = "\n".join(
        f"Scenario: {scenario}\nReport: {examples_df[examples_df.scenario == scenario].sample(1).log.iloc[0]}"
        for scenario in selected_scenarios
    )

    # Selecting a new scenario not in the examples
    scenario_mask = ~scenarios_df.incident_type.isin(selected_scenarios)
    if scenario_mask.sum() == 0:
        raise ValueError("No new scenarios available to select")

    scenario = scenarios_df[scenario_mask].sample().incident_type.iloc[0]

    prompt = (
        "You are an assistant tasked with generating example notes of calls to the "
        "emergency services, from the perspective of the call-handler/dispatcher. "
        "The reports are in a crude note-taking format, not full sentences/proper "
        "grammar. The following are examples of the desired outputs:\n"
        f"{scenarios_reports}\n"
        "Produce a short section from the middle of a report like the example using "
        f"the following scenario. {additional_instruction}\n\n"
        f"Scenario: {scenario}\n\n"
        "Report:"
    )

    return prompt, scenario


def display_formatted_output(prompt, responses, dark_theme=True):
    """
    Displays a prompt and its corresponding responses in a formatted HTML layout.

    The prompt and each response are enclosed in a styled HTML div element. The prompt is
    distinguished by a header, and each response is preceded by a numbered header. The function
    supports both a dark and light theme, determined by the dark_theme parameter.

    Parameters
    ----------
    prompt : str
        The input prompt that was given to the language model.
    responses : list of str
        A list of responses generated by the language model based on the prompt.
    dark_theme : bool, optional
        If True, the output will be displayed using a dark theme. Defaults to True.

    Returns
    -------
    None
        This function does not return any value. It renders HTML output directly.
    """

    # Styling options
    container_style = "margin: 20px; font-family: Arial, sans-serif;"
    prompt_style = "background-color: #e7f1ff; color: black; padding: 15px; border-radius: 10px; border: 1px solid #4F8BF9; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 10px;"
    response_style = "background-color: #f9f9f9; color: black; padding: 15px; border-radius: 10px; border: 1px solid #ccc; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-top: 10px;"
    prompt_header_style = "color: #3366ff; margin-bottom: 10px;"

    response_header_style = "color: white; margin-bottom: 0px; font-size: 18px" if dark_theme else "color: black; margin-bottom: 0px; font-size: 18px"

    formatted_prompt = prompt.replace("\n", "<br>")
    html_elements = [
        f"""
        <div style="{container_style}">
            <h2 style="{prompt_header_style}">Prompt:</h2>
            <div style="{prompt_style}">
                {formatted_prompt}
            </div>
        """
    ]

    for i, response in enumerate(responses, start=1):
        formatted_response = response.replace("\n", "<br>")
        response_block = f"""
            <h2 style="{response_header_style}">Response {i}:</h2>
            <div style="{response_style}">
                {formatted_response}
            </div>
        """
        html_elements.append(response_block)

    html_elements.append("</div>")
    full_html = ''.join(html_elements)
    display(HTML(full_html))


class OpenAIClient:
    """
    A client for interacting with OpenAI's GPT models.

    Attributes
    ----------
    api_key : str
        The API key used for OpenAI API requests.
    client : openai.OpenAI
        The OpenAI client instance.

    Methods
    -------
    get_gpt_model_completions(model, prompt, inf_params, labels, max_retries, retry_delay)
        Attempts to get model completions with retries on failure.
    """

    def __init__(self, api_key_file_path: str, save_path: str, timeout: int = 10):
        self.api_key = self._read_api_key(api_key_file_path)
        self.save_path = save_path
        self.client = openai.OpenAI(api_key=self.api_key, timeout=timeout)

    def _read_api_key(self, file_path: str) -> str:
        """
        Reads the OpenAI API key from a specified file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"API key file not found at {file_path}")
        with open(file_path, 'r') as file:
            return file.read().strip()

    def _get_chat_completions(self, model: str, messages: list, inf_params: dict,  
                              max_retries: int, retry_delay: int) -> list:
        """
        Attempts to get model completions with retries on failure.
        """
        for attempt in range(max_retries):
            try:
                chat_completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **inf_params)
                responses = [choice.message.content for choice in chat_completion.choices]
                return responses
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception("Maximum retries reached.") from e
                print(f"Exception: {e}\nRetrying... (Attempt {attempt + 1} of {max_retries})")
                time.sleep(retry_delay)

    def _save_responses(self, prompt: str, responses: list, labels: dict,
                        model_name: str, model_params: dict):
        """
        Saves the responses using the save_prompt_responses function.
        """
        save_prompt_responses(prompt, responses, labels, model_name,
                              self.save_path, model_params)

    def get_simple_model_completions(self, model: str, prompt: str,
                                  inf_params: dict, labels: dict = None,
                                  max_retries: int = 3, retry_delay: int = 5) -> list:
        """
        Generates simple response(s) to single prompt.

        Parameters
        ----------
        model : str
            The model identifier to use for completions.
        prompt : str
            The prompt to send to the model.
        inf_params : dict
            Additional inference parameters for the model.
        labels: dict, optional
            Any labels that apply to the generated examples
        max_retries : int, optional
            Maximum number of retries on failure (default is 3).
        retry_delay : int, optional
            Delay in seconds between retries (default is 5).

        Returns
        -------
        list
            A list of completion strings returned by the model.

        Raises
        ------
        Exception
            If the maximum number of retries is reached.
        """
        if labels is None:
            labels = {}

        messages = [{"role": "user", "content": prompt}]
        responses = self._get_chat_completions(model, messages, inf_params,
                                               max_retries, retry_delay)
        self._save_responses(prompt, responses, labels, model, inf_params)
        return responses


    def get_sequential_model_completions(self, model: str, initial_prompt: str,
                                         follow_up_prompt: str, inf_params: dict,
                                         labels: dict = None, n: int = 3,
                                         max_retries: int = 3, retry_delay: int = 5) -> list:
        """
        Gets sequential responses to initial prompt and repeated follow up prompts
        
        Intended to prompt model to produce multiple responses to the same request
        that differ from one another.

        Parameters
        ----------
        model : str
            The model identifier to use for completions.
        initial_prompt : str
            The initial prompt to send to the model.
        follow_up_prompt : str
            The follow-up prompt to send for subsequent responses.
        inf_params : dict
            Additional inference parameters for the model.
        labels: dict, optional
            Any labels that apply to the generated examples.
        sequence_length : int, optional
            Number of sequential responses to generate (default is 2).
        max_retries : int, optional
            Maximum number of retries on failure (default is 3).
        retry_delay : int, optional
            Delay in seconds between retries (default is 5).

        Returns
        -------
        list
            A list of sequential completion strings returned by the model.

        Raises
        ------
        Exception
            If the maximum number of retries is reached.
        """
        if labels is None:
            labels = {}
        if "n" in inf_params.keys():
            inf_params["n"] = 1
            print("Warning: Setting n to 1 - multiple responses not supported by sequential completions")

        messages = [{"role": "user", "content": initial_prompt}]
        responses = []
        for i in range(n):
            response = self._get_chat_completions(model, messages, inf_params,   
                                                  max_retries, retry_delay)[0]
            responses.append(response)
            response_message = {"role": "assistant", "content": response}
            follow_up_message = {"role": "user", "content": follow_up_prompt}
            messages += [response_message, follow_up_message]

        prompt = initial_prompt + f"\n\n[SEQUENTIAL PROMPT: {follow_up_prompt}]"  
        self._save_responses(prompt, responses, labels, model, inf_params)
        return responses
