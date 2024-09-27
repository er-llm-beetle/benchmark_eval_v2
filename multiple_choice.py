
# from rich import print
# import pandas as pd
# import ollama
# import time
# import os

from base import *


def get_model_answer_multiple_options(question, options, model, dstype) -> str:
    """
    Sends a query to the model and retrieves the response.

    Args:
        client: The Ollama client instance.
        question (str): The question to be categorized.
        options (str): The options for categorization.
        existing_answer (str): The existing answer.

    Returns:
        str: The model's response.
    """

    # client = ollama.Client()



    if dstype == 'tc':
        try:
            formatted_options = options.replace(", ", ",\n")
            options = formatted_options
        except Exception as e:
            print(f"No options - Error: {e}")
            options = 'No options - Error'

        messages = [
            {
                "role": "system",
                "content": "You are given a statement along with multiple options that represent different topics. Choose the option that best categorizes the statement based on its topic. Respond with only the letter of the chosen answer (e.g., A, B, C, etc.), without any additional text."
            },
            {
                "role": "user",
                "content": "Question:\nMənə yaxın filial tapmağa kömək edin.\nOptions:\nA) ATM,\nB) FEES,\nC) OTHER,\nD) CARD,\nE) ACCOUNT,\nF) TRANSFER,\nG) PASSWORD,\nH) LOAN,\nI) CONTACT,\nJ) FIND\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "J"
            },
            {
                "role": "user",
                "content": "Question:\nŞifrə təyin etməliyəm, bunu necə edə bilərəm?\nOptions:\nA) ATM,\nB) FEES,\nC) OTHER,\nD) CARD,\nE) ACCOUNT,\nF) TRANSFER,\nG) PASSWORD,\nH) LOAN,\nI) CONTACT,\nJ) FIND\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "G"
            },
            {
                "role": "user",
                "content": "Question:\nMən virtual kart almaq istəyirəm. Necə əldə edə bilərəm?\nOptions:\nA) ATM,\nB) FEES,\nC) OTHER,\nD) CARD,\nE) ACCOUNT,\nF) TRANSFER,\nG) PASSWORD,\nH) LOAN,\nI) CONTACT,\nJ) FIND\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "D"
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\nOptions:\n{options}"
            }
        ]

    # elif dstype == 'mc' or dstype == 'arc': # v1
    elif dstype == 'mc': # v2

        try:
            # if dstype == 'arc':
            #     formatted_options = ',\n'.join([f"{chr(65 + i)}) {sentence.rstrip('.')}" for i, sentence in enumerate(options)])
            #     print(formatted_options)
            #     options = formatted_options

            if dstype == 'mc':
                formatted_options = options.replace(", ", ",\n")
                options = formatted_options
        except Exception as e:
            print(f"No options - Error: {e}")
            options = 'No options - Error'

        messages = [
            {
                "role": "system",
                "content": "You are an AI that selects the most accurate answer in Azerbaijani based on a given question. You will be provided with a question in Azerbaijani and multiple options in Azerbaijani. Choose the single letter (A, B, C, D) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
            },
            {
                "role": "user",
                "content": "Question:\nBank krediti haqqında deyilənlərdən hansı yalnışdır?\nOptions:\nA) Bağlanmış müqaviləyə uyğun olaraq qaytarılmalıdır,\nB) Müddətin uzaldılması şərtinin müqavilədə olması zəruridir,\nC) Təminatsız verilə bilər,\nD) Pul vəsaitinin verilməsinə dair qarantiya və zəmanət öhdəlikləri kredit anlayışına aid deyil.\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "D"
            },
            {
                "role": "user",
                "content": "Question:\nDepozit haqqında deyilənlərdən hansı yalnışdır?\nOptions:\nA) Depozit müqaviləsi 0 (sıfır) faiz dərəcəsi ilə bağlana bilməz,\nB) Tələbli və ya müddətli ola bilər,\nC) Cari hesabda saxlanılan vəsait də depozit sayılır,\nD) Müştərinin tələbi ilə qaytarılmalıdır.\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Question:\nİpoteka krediti ilə bağlı hansı ifadə doğrudur?\nOptions:\nA) İpoteka krediti mütləq əmlak zəmanəti ilə verilir,\nB) İpoteka krediti yalnız maşın almaq üçün istifadə edilə bilər,\nC) İpoteka krediti heç bir faiz ödənilmədən verilə bilər,\nD) İpoteka krediti yalnız banklar tərəfindən təqdim olunur.\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\nOptions:\n{options}"
            }
        ]
    
    elif dstype == 'arc': # v2

        try:
            formatted_options = ',\n'.join([f"{chr(65 + i)}) {sentence.rstrip('.')}" for i, sentence in enumerate(options)])
            print(formatted_options)
            options = formatted_options

        except Exception as e:
            print(f"No options - Error: {e}")
            options = 'No options - Error'
    
        messages = [
            {
                "role": "system",
                "content": "You are an AI designed to answer questions in Azerbaijani based on reasoning and knowledge. You are an AI tasked with selecting the most accurate answer in Azerbaijani based on a given question.  Choose the single letter (A, B, C, D) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
            },
            {
                "role": "user",
                "content": "Question:\nDuzları sudan ayırmağın ən yaxşı yolu\nOptions:\nA) yağ,\nB) istilik,\nC) maqnit,\nD) əsəbləşmə alkoqolu\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "Question:\nŞagirdlər sinifdəki bir tankda bəzi cırtdan saxlayırdılar. Tank gündə bir neçə saat günəş işığı alacaq bir ərazidə saxlanıldı. Tankın bir hissəsində su, digər hissəsində isə bitkilərin böyüdüyü torpaq var idi.. Cırtdanlar bütün enerjisini haradan alırdılar?\nOptions:\nA) torpaq,\nB) su,\nC) bitkilər,\nD) günəş işığı\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "Question:\nBir tələbə təcrübəni təkrarlaya bilmək üçün hansı addımı atmalıdır?\nOptions:\nA) təcrübəni bir neçə dəfə təkrarlayın,\nB) bütün prosedurların qeydini saxlayır,\nC) Təcrübə quruluşunun bir modelini hazırlayın,\nD) Təcrübə şərtlərinin bir hissəsini dəyişdirin\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\nOptions:\n{options}"
            }
        ]


    else:
        raise("Invalid dstype")
    
    answer = ''
    try:
        stream = client_ollama.chat(
            model=model,  
            messages=messages,
            stream=True
        )
    except Exception as e:
        print(f"Error during streaming: {e}")
        return "Error"

    try:
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                answer += chunk['message']['content']
            else:
                print(f"Unexpected chunk format: {chunk}")
    except Exception as e:
        print(f"Error processing stream: {e}")
        return "Error"
    
    return answer


def compare_answers(actual_answer: str, predicted_answer: str) -> int:
    """
    Compare the actual answer with the predicted answer.
    
    Parameters:
    - actual_answer (str): The correct answer.
    - predicted_answer (str): The answer predicted by the model.
    
    Returns:
    - int: 1 if the answers match, otherwise 0.
    """
    # return 1 if actual_answer.lower() == predicted_answer.lower() else 0 # v1
    
    print("actual_answer:", actual_answer)
    print("predicted_answer:", predicted_answer)

    matched_predicted = re.match(r'[^A-Z]*([A-Z])', predicted_answer) # v2

    if matched_predicted.group(1) is not None and matched_predicted.group(1) != "":
        print("matched_predicted:", matched_predicted.group(1).lower())
        return 1 if actual_answer.lower() == matched_predicted.group(1).lower() else 0
    else:
        return 0
