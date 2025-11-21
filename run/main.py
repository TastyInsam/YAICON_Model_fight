import converse as c
import sys


def main():
    n = int(sys.argv[1])
    topic = sys.argv[2]
    init_persona_name = sys.argv[3]
    target_persona_name = sys.argv[4]
    init_persona = f"data/personas/{init_persona_name}.json"
    target_persona = f"data/personas/{target_persona_name}.json"
    c.agent_chat(n, topic, init_persona, target_persona)

if __name__ == "__main__":
    main()