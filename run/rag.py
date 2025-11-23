
import json, uuid, os, torch
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import openai
import re
from dotenv import load_dotenv
# from google.colab import userdata

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ### DB 구축하기

# In[3]:


embedding_model = SentenceTransformer('dragonkue/BGE-m3-ko')


# In[ ]:

DB_PATH = "qdrant_bge"
COLLECTION_NAME = "dcinside"

# 인덱스 생성하기
def database_indexing(client):
    data_path = "merged_dataset.json"

    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    points_to_upsert = []
    batch_size = 100

    for idx, item in enumerate(dataset):
        main_text = item.get('main', '')
        comments = item.get('comments', [])
        comments_text = ' '.join(comments) if comments else ''

        full_text = f"{main_text} {comments_text}".strip()

        if not full_text:
            continue

        vector = embedding_model.encode(full_text).tolist()

        payload = {
            "date": item.get('date', ''),
            "main": main_text,
            "comments": comments,
            "source_url": item.get('source_url', ''),
            "gallery": item.get('gallery', ''),
            "full_text": full_text
        }

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload
        )
        points_to_upsert.append(point)

        if len(points_to_upsert) >= batch_size:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points_to_upsert,
                wait=True
            )
            print(f"Indexed {idx + 1}/{len(dataset)} items...")
            points_to_upsert = []

    if points_to_upsert:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_to_upsert,
            wait=True
        )

    print(f"index completed : {len(dataset)}")

    return client


def database_check():
    client = QdrantClient(path=DB_PATH)

    collections = client.get_collections()
    collection_names = [c.name for c in collections.collections]

    if COLLECTION_NAME in collection_names:
        print(f" existing collection '{COLLECTION_NAME}'")
    else:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        print(f" new collection '{COLLECTION_NAME}' ")
        client = database_indexing(client)
    return client





# In[ ]:


# openai.api_key = userdata.get('OPENAI_KEY')


# ### RAG 함수 정의

# In[8]:


def search_documents(client, query: str, top_k: int = 3, gallery_filter: str = None) -> List[Dict]:
    query_vector = embedding_model.encode(query).tolist()

    query_filter = None
    if gallery_filter:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="gallery",
                    match=MatchValue(value=gallery_filter)
                )
            ]
        )

    search_response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True
    )

    results = []
    for hit in search_response.points:
        results.append({
            "score": hit.score,
            "date": hit.payload.get('date', ''),
            "main": hit.payload.get('main', ''),
            "comments": hit.payload.get('comments', []),
            "source_url": hit.payload.get('source_url', ''),
            "gallery": hit.payload.get('gallery', ''),
            "full_text": hit.payload.get('full_text', '')
        })

    return results


# In[9]:


def _prepare_rag_context(client, query: str, top_k: int = 3, gallery_filter: str = None) -> tuple:
    retrieved_docs = search_documents(client, query, top_k=top_k, gallery_filter=gallery_filter)

    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"[문서 {i}]")
        context_parts.append(f"날짜: {doc['date']}")
        context_parts.append(f"갤러리: {doc['gallery']}")
        context_parts.append(f"내용: {doc['main']}")
        if doc['comments']:
            context_parts.append(f"댓글: {', '.join(doc['comments'][:3])}")
        context_parts.append("")

    context = "\n".join(context_parts)
    return retrieved_docs, context


# api로 데려오기

# In[10]:


def generate_rag_response_api(
    query: str,
    model: str = "gpt-4o-mini",
    top_k: int = 3,
    gallery_filter: str = None,
    max_tokens: int = 512,
    temperature: float = 0.7
) -> Dict[str, Any]:

    retrieved_docs, context = _prepare_rag_context(query, top_k, gallery_filter)
    prompt = f"""다음 문서에 있는대로 질문에 답변해주세요.

{context}

질문: {query}

답변:"""

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "당신은 제공된 문서에 있는대로 답변하는 도우미입니다."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    answer = response.choices[0].message.content.strip()

    return {
        "query": query,
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "model_used": model
    }


# In[ ]:


# get_ipython().system('ls -lh ~/.cache/huggingface/hub | grep models')


# 로컬 모델로 데려오기

# In[14]:

def model_setup(mode):
    load_dotenv()
    if mode == "base": model_name = os.getenv("BASE_MODEL_NAME")
    elif mode == "detox": model_name = os.getenv("DETOX_MODEL_NAME")
    else:
        print("wrong mode selection. Choose between 0 and 1.")
        return
    
    print(model_name)

    # 양자화 필요할 시 (코랩 기준 8B 이상)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=quantization_config, # 필요할 시
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        # device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True
    ).to(device)

    print('load done !')
    
    return tokenizer, model


# In[ ]:


def generate_rag_response_local(
    tokenizer,
    model,
    client,
    summary,
    query: str,
    top_k: int = 10,
    gallery_filter: str = None,
    max_tokens: int = 150,
    temperature: float = 0.45,
) -> Dict[str, Any]:

    retrieved_docs, _ = _prepare_rag_context(client, query, top_k, gallery_filter)

    print(f"\n🔍 검색된 원본 문서 개수: {len(retrieved_docs)}개")

    # 2. 중복 제거 및 내용 병합
    unique_docs = []
    seen_contents = set()

    for doc in retrieved_docs:
        # 본문과 댓글을 모두 합쳐서 풍부한 Context 만들기
        content_parts = [doc['main'].strip()]
        if doc['comments']:
            # 댓글도 내용에 포함 (너무 짧은 본문 보완)
            content_parts.extend(doc['comments'][:2])
        
        full_content = " ".join(content_parts)
        
        # 내용이 너무 짧거나 중복이면 패스, 아니면 추가
        if len(full_content) > 2 and full_content not in seen_contents:
            unique_docs.append(doc)
            seen_contents.add(full_content)

    # [디버깅] 중복 제거 후 개수 출력
    print(f"✨ 중복 제거 후 남은 문서: {len(unique_docs)}개")

    # Context 구성
    context_list = []
    for doc in unique_docs:
        # 본문 + 댓글 결합
        text = doc['main']
        if doc['comments']:
            text += " (" + " ".join(doc['comments'][:2]) + ")"
        context_list.append(text)
        
    context = "\n".join(context_list)

    print("\n" + "="*20 + " [Context Preview] " + "="*20)
    print(context[:500] + "..." if len(context) > 500 else context)
    print("="*60 + "\n")

    # 문서가 없으면 바로 리턴
    if not context.strip():
        return {
            "query": query,
            "answer": "관련된 내용이 DB에 없음.",
            "retrieved_docs": []
        }

    print(f"\n🔍 검색된 원본 문서 개수: {len(retrieved_docs)}개")

    # Prompt 구성
    prompt = f"""
#Constraints. 
##1. Answer the question based ONLY on the provided Documents below.
##2. Do not make up information.
##3. you MUST answer in **KOREAN**.
##4. Output a string of the following format: 
    - "your answer ONLY"
- ex1) 
    "여가부같은 소리하고 앉았노ㅋㅋ 절대 안된다 게이야ㅋㅋㅋㅋㅋ"
- ex2)
    "모병제는 씨발 좆같은 소리하고 앉았네 ㄹㅇ"


#Documents:       
-{context}

#Question: 
-{summary}

Answer:
  """
    print("\n")
    print("====final prompt====")
    print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.8,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty = 1.3
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text.split("Answer:")[-1].strip()

    return answer


def generate_rag_response_local_old(
    tokenizer,
    model,
    client,
    query: str,
    top_k: int = 10,
    gallery_filter: str = None,
    max_tokens: int = 150,
    temperature: float = 0.45,
) -> Dict[str, Any]:

    print("="*50)
    print('\n')
    print("query: ")
    print(query)

    retrieved_docs, _ = _prepare_rag_context(client, query, top_k, gallery_filter)

    print(f"\n🔍 검색된 원본 문서 개수: {len(retrieved_docs)}개")

    # 2. 중복 제거 및 내용 병합
    unique_docs = []
    seen_contents = set()

    for doc in retrieved_docs:
        # 본문과 댓글을 모두 합쳐서 풍부한 Context 만들기
        content_parts = [doc['main'].strip()]
        if doc['comments']:
            # 댓글도 내용에 포함 (너무 짧은 본문 보완)
            content_parts.extend(doc['comments'][:2])
        
        full_content = " ".join(content_parts)
        
        # 내용이 너무 짧거나 중복이면 패스, 아니면 추가
        if len(full_content) > 2 and full_content not in seen_contents:
            unique_docs.append(doc)
            seen_contents.add(full_content)

    # [디버깅] 중복 제거 후 개수 출력
    print(f"✨ 중복 제거 후 남은 문서: {len(unique_docs)}개")

    # Context 구성
    context_list = []
    for doc in unique_docs:
        # 본문 + 댓글 결합
        text = doc['main']
        if doc['comments']:
            text += " (" + " ".join(doc['comments'][:2]) + ")"
        context_list.append(text)
        
    context = "\n".join(context_list)

    print("\n" + "="*20 + " [Context Preview] " + "="*20)
    print(context[:500] + "..." if len(context) > 500 else context)
    print("="*60 + "\n")

    # 문서가 없으면 바로 리턴
    if not context.strip():
        return {
            "query": query,
            "answer": "관련된 내용이 DB에 없음.",
            "retrieved_docs": []
        }
    
    # Prompt 구성
    messages = [
        {
            "role": "system",
            "content": (
                "너는 디시인사이드 갤러리 유저다. "
                "주어진 [텍스트]들의 내용을 종합해서 **반말(비속어, 음슴체)**로 한 문장으로 요약해라. "
                "니 생각은 넣지 말고, 텍스트에 있는 여론만 그대로 읊어라."
            )
        },
        {
            "role": "user",
            "content": (
                f"아래 [텍스트] 읽고 디시 말투로 한 줄 요약해.\n"
                f"내용:\n{context}\n\n"
                f"요약:"
            )
        }
    ]

    print("="*50)
    print('\n')
    print("FINAL PROMPT: ")
    print(messages)
    


    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            eos_token_id = terminators,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty = 1.1,
            no_repeat_ngram_size = 0
        )

    response = outputs[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens = True).strip()

    # 후처리
    if "요약" in answer:
        answer = answer.split("요약:")[-1].strip()
        
    # 불필요한 기호/영어 제거 (한글, 숫자, 문장부호만)
    answer = re.sub(r'[^가-힣0-9\s.,!?~ㅋㅎ]', '', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()

    return {
        "query": query,
        "answer": answer,
        "retrieved_docs": unique_docs
    }



# client = database_check()
# tokenizer, model = model_setup("base")
# answer = generate_rag_response_local_old(tokenizer, model, client, "여성가족부를 페지해야 할까?" )
# print("response: ")
# print(answer)



# In[ ]:


def interactive_rag():
    while True:
        user_input = input("\n질문: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            print("\n대화를 종료합니다.")
            break


# In[ ]:


