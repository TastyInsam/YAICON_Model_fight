import utils
import model as m

# def generate_agent_chat(init_persona,
#                         target_persona,
#                         curr_context,
#                         init_idea,
#                         target_idea):
#     summarized_idea = m.run_model_agent_chat(init_persona,
#                                             target_persona,
#                                             curr_context,
#                                             init_idea,
#                                             target_idea
#                                            )
    
def generate_one_utterance(topic,
                           init_persona, 
                           target_persona, 
                           context, 
                           curr_chat):
    # curr_context = (f"This is your persona info.\n" +
    #                 f"name: {init_persona.name}\n" +
    #                 f"age: {init_persona.age}\n" +
    #                 f"gender: {init_persona.gender}\n" + 
    #                 f"religion: {init_persona.religion}\n" +
    #                 f"race: {init_persona.race}\n\n" +
    #                 f"Topic: {init_persona.topic}\n" +
    #                 f"Main Stance: {init_persona.stance}\n" +
    #                 f"Background: {init_persona.background}\n")
    # curr_context += ("You are")

    x = m.run_model_generate_chat_utt(topic, init_persona, target_persona, curr_context, curr_chat)[0]
    


                    
                    
    


def agent_chat(n, topic, init_persona, target_persona):
    # create 
    curr_chat= []

    for i in range(n):
        context = "" # context = persona(main stance) + chat history

        # chat_str = ""
        # for i in curr_chat:
        #     chat_str += ": ".join(i) + "\n"
        
        utt, end = generate_one_utterance(topic,init_persona, target_persona, context, curr_chat)

        curr_chat += [[init_persona.name, utt]]
        if end:
            break

        context  = "" # context = target_persona

        # for i in curr_chat:
        #     chat_str += ": ".join(i) + "\n"
        
        utt, end = generate_one_utterance(topic, target_persona, init_persona, context, curr_chat)
        
        curr_chat += [[target_persona.name, utt]]

        if end:
            break
    
    d = []
    d.append({
        "meta" : ""
    })
    for row in curr_chat:
        d.append({
            f"{row[0]}" : f"{row[1]}"
        })

    utils.record_json(d, trajectory_path)
    