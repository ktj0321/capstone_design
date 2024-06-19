from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 데이터 불러오기
url = "https://raw.githubusercontent.com/ktj0321/capstone_design/main/recipe_data.csv"
df = pd.read_csv(url)
# 오류를 일으키는 레시피 제거
df.drop(947, inplace=True)

# 한글 문자만 추출하는 정규 표현식 패턴
pattern = re.compile(r'[가-힣]+')


def extract_ingredients(text):
  # 정규 표현식을 사용하여 재료 이름 추출
  ingredients = pattern.findall(text)
  # 중복 제거
  ingredients = list(set(ingredients))
  return ingredients


def calculate_cosine_similarity(user_ingredients, recipe_ingredients):
  # 모든 재료를 하나의 리스트로 합침
  all_ingredients = [user_ingredients, recipe_ingredients]

  # CountVectorizer를 사용하여 재료를 벡터화
  vectorizer = CountVectorizer().fit(all_ingredients)
  vectors = vectorizer.transform(all_ingredients).toarray()

  # 코사인 유사도 계산
  similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

  return similarity


def recommend_recipes(user_ingredients, df):
  # DataFrame의 각 행에 대해 사용자 입력과의 코사인 유사도 계산
  df['cosine_similarity'] = df['ingredients'].apply(
      lambda x: calculate_cosine_similarity(user_ingredients, x))

  # 코사인 유사도에 따라 DataFrame 정렬
  df_sorted = df.sort_values(by='cosine_similarity', ascending=False)

  # 상위 6개 레시피 선택
  top_6_recipes = df_sorted.head(6)

  return top_6_recipes


@app.route('/')
def home():
  return render_template('index.html')


@app.route('/add_ingredient')
def add_ingredient():
  return render_template('add_ingredient.html')


@app.route('/add_ingredient_txt')
def add_ingredient_txt():
  return render_template('add_ingredient_txt.html')


@app.route('/recommended_recipe', methods=["get"])
def recommended_recipe():
  ingredient = request.args.get('ingredient')
  result = recommend_recipes(ingredient, df)
  rcp_name_1 = result.iloc[0]["RCP_NM"]
  rcp_name_2 = result.iloc[1]["RCP_NM"]
  rcp_name_3 = result.iloc[2]["RCP_NM"]
  rcp_name_4 = result.iloc[3]["RCP_NM"]
  rcp_name_5 = result.iloc[4]["RCP_NM"]
  rcp_name_6 = result.iloc[5]["RCP_NM"]
  rcp_description_1 = result.iloc[0]["RCP_NA_TIP"]
  rcp_description_2 = result.iloc[1]["RCP_NA_TIP"]
  rcp_description_3 = result.iloc[2]["RCP_NA_TIP"]
  rcp_description_4 = result.iloc[3]["RCP_NA_TIP"]
  rcp_description_5 = result.iloc[4]["RCP_NA_TIP"]
  rcp_description_6 = result.iloc[5]["RCP_NA_TIP"]
  ingredients_recipe1 = result.iloc[0]["RCP_PARTS_DTLS"]
  ingredients_recipe2 = result.iloc[1]["RCP_PARTS_DTLS"]
  ingredients_recipe3 = result.iloc[2]["RCP_PARTS_DTLS"]
  ingredients_recipe4 = result.iloc[3]["RCP_PARTS_DTLS"]
  ingredients_recipe5 = result.iloc[4]["RCP_PARTS_DTLS"]
  ingredients_recipe6 = result.iloc[5]["RCP_PARTS_DTLS"]
  kcal_recipe1 = result.iloc[0]["INFO_ENG"]
  kcal_recipe2 = result.iloc[1]["INFO_ENG"]
  kcal_recipe3 = result.iloc[2]["INFO_ENG"]
  kcal_recipe4 = result.iloc[3]["INFO_ENG"]
  kcal_recipe5 = result.iloc[4]["INFO_ENG"]
  kcal_recipe6 = result.iloc[5]["INFO_ENG"]
  car_recipe1 = result.iloc[0]["INFO_CAR"]
  car_recipe2 = result.iloc[1]["INFO_CAR"]
  car_recipe3 = result.iloc[2]["INFO_CAR"]
  car_recipe4 = result.iloc[3]["INFO_CAR"]
  car_recipe5 = result.iloc[4]["INFO_CAR"]
  car_recipe6 = result.iloc[5]["INFO_CAR"]
  pro_recipe1 = result.iloc[0]["INFO_PRO"]
  pro_recipe2 = result.iloc[1]["INFO_PRO"]
  pro_recipe3 = result.iloc[2]["INFO_PRO"]
  pro_recipe4 = result.iloc[3]["INFO_PRO"]
  pro_recipe5 = result.iloc[4]["INFO_PRO"]
  pro_recipe6 = result.iloc[5]["INFO_PRO"]
  fat_recipe1 = result.iloc[0]["INFO_FAT"]
  fat_recipe2 = result.iloc[1]["INFO_FAT"]
  fat_recipe3 = result.iloc[2]["INFO_FAT"]
  fat_recipe4 = result.iloc[3]["INFO_FAT"]
  fat_recipe5 = result.iloc[4]["INFO_FAT"]
  fat_recipe6 = result.iloc[5]["INFO_FAT"]
  na_recipe1 = result.iloc[0]["INFO_NA"]
  na_recipe2 = result.iloc[1]["INFO_NA"]
  na_recipe3 = result.iloc[2]["INFO_NA"]
  na_recipe4 = result.iloc[3]["INFO_NA"]
  na_recipe5 = result.iloc[4]["INFO_NA"]
  na_recipe6 = result.iloc[5]["INFO_NA"]
  rcp_step_recipe1_1 = result.iloc[0]["MANUAL01"]
  rcp_step_recipe1_2 = result.iloc[0]["MANUAL02"]
  rcp_step_recipe1_3 = result.iloc[0]["MANUAL03"]
  rcp_step_recipe1_4 = result.iloc[0]["MANUAL04"]
  rcp_step_recipe1_5 = result.iloc[0]["MANUAL05"]
  rcp_step_recipe1_6 = result.iloc[0]["MANUAL06"]
  rcp_step_recipe2_1 = result.iloc[1]["MANUAL01"]
  rcp_step_recipe2_2 = result.iloc[1]["MANUAL02"]
  rcp_step_recipe2_3 = result.iloc[1]["MANUAL03"]
  rcp_step_recipe2_4 = result.iloc[1]["MANUAL04"]
  rcp_step_recipe2_5 = result.iloc[1]["MANUAL05"]
  rcp_step_recipe2_6 = result.iloc[1]["MANUAL06"]
  rcp_step_recipe3_1 = result.iloc[2]["MANUAL01"]
  rcp_step_recipe3_2 = result.iloc[2]["MANUAL02"]
  rcp_step_recipe3_3 = result.iloc[2]["MANUAL03"]
  rcp_step_recipe3_4 = result.iloc[2]["MANUAL04"]
  rcp_step_recipe3_5 = result.iloc[2]["MANUAL05"]
  rcp_step_recipe3_6 = result.iloc[2]["MANUAL06"]
  rcp_step_recipe4_1 = result.iloc[3]["MANUAL01"]
  rcp_step_recipe4_2 = result.iloc[3]["MANUAL02"]
  rcp_step_recipe4_3 = result.iloc[3]["MANUAL03"]
  rcp_step_recipe4_4 = result.iloc[3]["MANUAL04"]
  rcp_step_recipe4_5 = result.iloc[3]["MANUAL05"]
  rcp_step_recipe4_6 = result.iloc[3]["MANUAL06"]
  rcp_step_recipe5_1 = result.iloc[4]["MANUAL01"]
  rcp_step_recipe5_2 = result.iloc[4]["MANUAL02"]
  rcp_step_recipe5_3 = result.iloc[4]["MANUAL03"]
  rcp_step_recipe5_4 = result.iloc[4]["MANUAL04"]
  rcp_step_recipe5_5 = result.iloc[4]["MANUAL05"]
  rcp_step_recipe5_6 = result.iloc[4]["MANUAL06"]
  rcp_step_recipe6_1 = result.iloc[5]["MANUAL01"]
  rcp_step_recipe6_2 = result.iloc[5]["MANUAL02"]
  rcp_step_recipe6_3 = result.iloc[5]["MANUAL03"]
  rcp_step_recipe6_4 = result.iloc[5]["MANUAL04"]
  rcp_step_recipe6_5 = result.iloc[5]["MANUAL05"]
  rcp_step_recipe6_6 = result.iloc[5]["MANUAL06"]

  return render_template('recommended_recipe.html',
                         rcp_name_1=rcp_name_1,
                         rcp_name_2=rcp_name_2,
                         rcp_name_3=rcp_name_3,
                         rcp_name_4=rcp_name_4,
                         rcp_name_5=rcp_name_5,
                         rcp_name_6=rcp_name_6,
                         rcp_description_1=rcp_description_1,
                         rcp_description_2=rcp_description_2,
                         rcp_description_3=rcp_description_3,
                         rcp_description_4=rcp_description_4,
                         rcp_description_5=rcp_description_5,
                         rcp_description_6=rcp_description_6,
                         ingredients_recipe1=ingredients_recipe1,
                         ingredients_recipe2=ingredients_recipe2,
                         ingredients_recipe3=ingredients_recipe3,
                         ingredients_recipe4=ingredients_recipe4,
                         ingredients_recipe5=ingredients_recipe5,
                         ingredients_recipe6=ingredients_recipe6,
                         kcal_recipe1=kcal_recipe1,
                         kcal_recipe2=kcal_recipe2,
                         kcal_recipe3=kcal_recipe3,
                         kcal_recipe4=kcal_recipe4,
                         kcal_recipe5=kcal_recipe5,
                         kcal_recipe6=kcal_recipe6, car_recipe1=car_recipe1,
                          car_recipe2=car_recipe2,
                          car_recipe3=car_recipe3,
                          car_recipe4=car_recipe4,
                          car_recipe5=car_recipe5,
                          car_recipe6=car_recipe6, pro_recipe1=pro_recipe1,
                          pro_recipe2=pro_recipe2,
                          pro_recipe3=pro_recipe3,
                          pro_recipe4=pro_recipe4,
                          pro_recipe5=pro_recipe5,
                          pro_recipe6=pro_recipe6, fat_recipe1=fat_recipe1,
                         fat_recipe2=fat_recipe2,
                         fat_recipe3=fat_recipe3,
                         fat_recipe4=fat_recipe4,
                         fat_recipe5=fat_recipe5,
                         fat_recipe6=fat_recipe6,

                         na_recipe1=na_recipe1,
                         na_recipe2=na_recipe2,
                         na_recipe3=na_recipe3,
                         na_recipe4=na_recipe4,
                         na_recipe5=na_recipe5,
                         na_recipe6=na_recipe6,
                         rcp_step_recipe1_1=rcp_step_recipe1_1,
                         rcp_step_recipe1_2=rcp_step_recipe1_2,
                         rcp_step_recipe1_3=rcp_step_recipe1_3,
                         rcp_step_recipe1_4=rcp_step_recipe1_4,
                         rcp_step_recipe1_5=rcp_step_recipe1_5,
                         rcp_step_recipe1_6=rcp_step_recipe1_6,
                         rcp_step_recipe2_1=rcp_step_recipe2_1,
                         rcp_step_recipe2_2=rcp_step_recipe2_2,
                         rcp_step_recipe2_3=rcp_step_recipe2_3,
                         rcp_step_recipe2_4=rcp_step_recipe2_4,
                         rcp_step_recipe2_5=rcp_step_recipe2_5,
                         rcp_step_recipe2_6=rcp_step_recipe2_6,
                         rcp_step_recipe3_1=rcp_step_recipe3_1,
                         rcp_step_recipe3_2=rcp_step_recipe3_2,
                         rcp_step_recipe3_3=rcp_step_recipe3_3,
                         rcp_step_recipe3_4=rcp_step_recipe3_4,
                         rcp_step_recipe3_5=rcp_step_recipe3_5,
                         rcp_step_recipe3_6=rcp_step_recipe3_6,
                         rcp_step_recipe4_1=rcp_step_recipe4_1,
                         rcp_step_recipe4_2=rcp_step_recipe4_2,
                         rcp_step_recipe4_3=rcp_step_recipe4_3,
                         rcp_step_recipe4_4=rcp_step_recipe4_4,
                         rcp_step_recipe4_5=rcp_step_recipe4_5,
                         rcp_step_recipe4_6=rcp_step_recipe4_6,
                         rcp_step_recipe5_1=rcp_step_recipe5_1,
                         rcp_step_recipe5_2=rcp_step_recipe5_2,
                         rcp_step_recipe5_3=rcp_step_recipe5_3,
                         rcp_step_recipe5_4=rcp_step_recipe5_4,
                         rcp_step_recipe5_5=rcp_step_recipe5_5,
                         rcp_step_recipe5_6=rcp_step_recipe5_6,
                         rcp_step_recipe6_1=rcp_step_recipe6_1,
                         rcp_step_recipe6_2=rcp_step_recipe6_2,
                         rcp_step_recipe6_3=rcp_step_recipe6_3,
                         rcp_step_recipe6_4=rcp_step_recipe6_4,
                         rcp_step_recipe6_5=rcp_step_recipe6_5,
                         rcp_step_recipe6_6=rcp_step_recipe6_6)


if __name__ == '__main__':
  app.run(debug=True)
