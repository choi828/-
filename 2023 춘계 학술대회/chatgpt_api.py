import openai
import csv
import pandas as pd
from fuzzywuzzy import fuzz
import time

API_KEY = 'API_KEY'
openai.api_key = API_KEY

# 입력 파일 경로
input_file_path = "/Users/daewoong/Documents/서울대학교/NLP/연구 및 프로젝트/pororo_ner/geo-test.txt"
# 출력 CSV 파일 경로
output_file_path = "/Users/daewoong/Documents/서울대학교/NLP/2023 춘계 학술대회/chatGPT_output.csv"

df = pd.read_csv(input_file_path)
print(df)
total_fuzzy_score = 0

for index, row in df.iterrows():
    question = row["NL_substituted"]
    print(question)

with open(output_file_path, "w", newline="") as output_file:
    writer = csv.writer(output_file)
    writer.writerow(["Question", "Predicted Entity"])

    for index, row in df.iterrows():
        question = row["NL_substituted"]
        # entity = row["entity_list"]

        # NER 수행    
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f'"{question}" 문장만 Named Entity Recognition 해줘 ORG, LOC 를 사용해줘'}
            ]
        )
        predicted_entities = response['choices'][0]['message']['content'].split("\n")
        predicted_entities_dict = {}
        for entity_info in predicted_entities:
            if ":" in entity_info:
                entity_type, entity_value = entity_info.split(":")
                entity_type = entity_type.strip()
                entity_value = entity_value.strip()
                predicted_entities_dict[entity_type] = entity_value
        print(predicted_entities)
        # fuzzy_scores = {}
        # for entity_type in ["PER", "ORG", "LOC"]:
        #     predicted_entity = predicted_entities_dict.get(entity_type, "없음")
        #     fuzzy_score = fuzz.ratio(entity, predicted_entity)
        #     fuzzy_scores[entity_type] = fuzzy_score
        # print(predicted_entities_dict)
        # 결과를 CSV 파일에 작성
        writer.writerow([question, predicted_entities_dict])
        time.sleep(20)

print("NER and evaluation completed. Output CSV file created.")
