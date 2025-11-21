import json
import model_structure as ms

def run_model_generate_chat_utt(topic,
                                init_persona, 
                                target_persona, 
                                context, 
                                curr_chat):
    def create_prompt_input(topic, init_persona, target_persona, curr_context, curr_chat):
        persona = init_persona

        persona_str = ""
        for key, val in persona:
            persona_str += f"{key}: {val}\n"

        curr_chat_str = ""
        for i in curr_chat:
            curr_chat_str += ": ".join(i) + "\n"
        if curr_chat_str == "":
            curr_chat_str = "The conversation has not started yet -- start it!]"

        init_description = f"Here is a brief description of {init_persona.name}.\n{persona_str}"
        prompt_input = [
            init_description,
            init_persona.name,
            curr_context, 
            init_persona.name, 
            target_persona.name,
            init_persona.name, 
            init_persona.name,
            init_persona.name]
        return prompt_input
    
    def __chat_func_clean_up(gpt_response, prompt=""): 
        gpt_response = extract_first_json_dict(gpt_response)

        cleaned_dict = dict()
        cleaned = []
        for key, val in gpt_response.items(): 
            cleaned += [val]
        cleaned_dict["utterance"] = cleaned[0]
        cleaned_dict["end"] = True
        if "f" in str(cleaned[1]) or "F" in str(cleaned[1]): 
            cleaned_dict["end"] = False

        return cleaned_dict

    def __chat_func_validate(gpt_response, prompt=""): 
        print ("ugh...")
        try: 
        # print ("debug 1")
        # print (gpt_response)
        # print ("debug 2")

            print (extract_first_json_dict(gpt_response))
            # print ("debug 3")
            return True
        except:
            return False 
        
    def get_fail_safe():
        cleaned_dict = dict()
        cleaned_dict["utterance"] = "..."
        cleaned_dict["end"] = False
        return cleaned_dict

    print("11")
    prompt_template = "data/prompt/conversation.txt"
    prompt_input = create_prompt_input(topic, init_persona, target_persona, context, curr_chat)
    
    print("22")
    prompt = generate_prompt(prompt_input, prompt_template)
    
    print(prompt)

    output = ms.get_model_request(prompt)

    print(output)
    param = {"engine": "text-davinci-003", "max_tokens": 50, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    return output, [output, prompt, param, prompt_input]

def generate_prompt(curr_input, prompt_lib_file): 
    """
    Takes in the current input (e.g. comment that you want to classifiy) and 
    the path to a prompt file. The prompt file contains the raw str prompt that
    will be used, which contains the following substr: !<INPUT>! -- this 
    function replaces this substr with the actual curr_input to produce the 
    final promopt that will be sent to the GPT3 server. 
    ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
    RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
    """
    if type(curr_input) == type("string"): 
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]

    f = open(prompt_lib_file, "r")
    prompt = f.read()
    f.close()
    for count, i in enumerate(curr_input):   
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt: 
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    return prompt.strip()

def extract_first_json_dict(data_str):
# Find the first occurrence of a JSON object within the string
    start_idx = data_str.find('{')
    end_idx = data_str.find('}', start_idx) + 1

    # Check if both start and end indices were found
    if start_idx == -1 or end_idx == 0:
        return None

    # Extract the first JSON dictionary
    json_str = data_str[start_idx:end_idx]

    try:
        # Attempt to parse the JSON data
        json_dict = json.loads(json_str)
        return json_dict
    except json.JSONDecodeError:
        # If parsing fails, return None
        return None
