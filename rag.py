
# load_dotenv()




# client = ollama.Client()


from __init__ import *


def create_combined_prompt_context(context: str, question: str) -> str:
    """
    Create the prompt for the LLM to generate an answer based on the given context and question.
    """

    # v1
    # return f"""
    #     You are an answer generator AI in Azerbaijani. Your task is to generate answers based on the provided context and the given question.

    #     **Example:**

    #     **Context:** Azərbaycan Respublikası Cənubi Qafqazda yerləşən bir ölkədir. İqtisadi və mədəni mərkəzi Bakı şəhəridir.
        
    #     **Question in Azerbaijani:** Azərbaycan Respublikasının paytaxtı haradır?

    #     **Generated Answer in Azerbaijani:** Bakı şəhəri.

    #     **Your Task:**

    #     **Context in Azerbaijani:** {context}

    #     **Question in Azerbaijani:** {question}

    #     Provide a clear and accurate answer in Azerbaijani based on the context, and include your answer in 1-2 sentences.
    # """

    # v1.1
    return (
        f"You are an answer generator AI in Azerbaijani. Your task is to generate answers based on the provided context and the given question.\n"
        f"Provide a clear and accurate answer in Azerbaijani based on the context, and include your answer in 1-2 sentences.\n\n"
        f"Context in Azerbaijani:\n"
        f"{context}\n\n"
        f"Question in Azerbaijani:\n"
        f"{question}\n\n"
        f"Answer:"
    )

    # # v2.1
    # return (
    #     f"Context in Azerbaijani:\n"
    #     f"{context}\n\n"
    #     f"Question in Azerbaijani:\n"
    #     f"{question}\n\n"
    #     f"Answer:"
    # )



def get_answer_from_local_ollama_context(model: str, question: str, context: str) -> str:
    """
    Send a prompt to the local Ollama model and retrieve the answer using the ollama library.
    """
    prompt = create_combined_prompt_context(context, question)

    # Prepare messages for v2
    system_message = "You are an answer generator AI in Azerbaijani. Your task is to generate answers based on the provided context and the given question.\nProvide a clear and accurate answer in Azerbaijani based on the context, and include your answer in 1-2 sentences.\n"
    messages = [
        # {'role': 'system', 'content': system_message},  # Set the system context # v2
        {'role': 'user', 'content': prompt}  # The user's prompt # v1, v2
    ]

    stream = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}], # v1
        # messages=messages, # v2
        stream=True
    )
    
    answer = ''
    try:
        for chunk in stream:
            answer += chunk['message']['content']
    except Exception as e:
        logging.error(f"Request to local Ollama failed: {e}")
    
    if len(answer) > 400:
        return 'Long answer'
    return answer.strip() if answer else "Error"

def get_evaluation_score_context(question: str, actual_answer: str, predicted_answer: str) -> str:
    """
    Generate an evaluation score between 0 and 100 by comparing the actual and predicted answers.
    """
    # httpx_client = httpx.Client(http2=True, verify=False)

    # Initialize OpenAI client for NVIDIA
    # client = OpenAI(base_url=BASE_URL_LLM, api_key=API_KEY_LLM, http_client=httpx_client)


    prompt = f"""
            Evaluate the following answers and provide a score from 0 to 100 based on how well the predicted
            answer matches the actual answer based on the asked question. Provide the score only, without any additional text.

            0-10: No answer or completely incorrect
            11-30: Significant errors or missing key information
            31-50: Some errors or incomplete information, but recognizable effort
            51-70: Mostly accurate with minor errors or omissions
            71-90: Very close to the actual answer with only minor discrepancies
            91-100: Accurate or nearly perfect match

            **Example:**

            **Question that asked in Azerbaijani:** Makroiqtisadiyyat nədir və mikroiqtisadiyyatdan necə fərqlənir?  
            **Actual Answer in Azerbaijani:** Makroiqtisadiyyat iqtisadiyyatın böyük miqyasda təhlili ilə məşğul olur, mikroiqtisadiyyat isə kiçik miqyasda, yəni fərdi bazarlarda və şirkətlərdə baş verən prosesləri öyrənir.  
            **Predicted Answer in Azerbaijani:** Makroiqtisadiyyat iqtisadiyyatın ümumi aspektlərini öyrənir, mikroiqtisadiyyat isə fərdi bazarları təhlil edir.  
            **Score (0 to 100):** 65

            **Your Task:**

            **Question that asked in Azerbaijani:** {question}

            **Actual Answer in Azerbaijani:** {actual_answer}

            **Predicted Answer in Azerbaijani:** {predicted_answer}

            **Score (0 to 100):**
            """


    payload = {
        "model": MODEL_LLAMA_3_1_405B,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 50
    }

    for attempt in range(NUM_RETRIES):
        try:
            completion = client_openai.chat.completions.create(**payload)
            if completion.choices:
                content = completion.choices[0].message.content
                if content:
                    return content.strip()
                logging.error("Content in response is None.")
            else:
                logging.error(f"Unexpected response format: {completion}")
        except Exception as e:
            logging.error(f"Request failed: {e}")
            # if attempt < NUM_RETRIES - 1:
            #     sleep_time = BASE_SLEEP_TIME * (2 ** attempt)
            #     logging.info(f"Retrying in {sleep_time} seconds...")
            #     time.sleep(sleep_time)
            # else:
            #     return "No score received"
    return "Error"

