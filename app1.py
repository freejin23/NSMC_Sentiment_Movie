from PIL import Image
import requests
import streamlit as st

from bs4 import BeautifulSoup as bs
from io import BytesIO
from urllib import request
import re
import datetime
import torch
from transformers import BertTokenizer
from time import sleep
from scrapper import NaverReviewScrapper


from model.model import BertClassifier, prediction


@st.cache
def load_model():
    model = BertClassifier("klue/bert-base", linear_size=256, num_class=1)
    model.load_state_dict(torch.load("model/model.pt", map_location="cpu"))
    model.eval()
    return model


@st.cache(allow_output_mutation=True)
def load_tokenizer():
    return BertTokenizer.from_pretrained("klue/bert-base", do_lower_case=False)


@st.cache
def make_year_option():
    date = datetime.date.today()
    year = date.strftime("%Y")
    options = ["전체"]
    options += [y for y in range(int(year), int(year) - 50, -1)]

    return options


def get_movie_poster(link):
    response = requests.get(link)
    return bs(response.content, "html.parser").find("img")["src"]


def get_movie_info(title, year=0):
    header_parms = {
        "X-Naver-Client-Id": "GRqRk1lDm03b11aXGKc9",
        "X-Naver-Client-Secret": "hWgGOZD2eB",
    }

    if year:
        url = f"https://openapi.naver.com/v1/search/movie.json?query={title}&yearfrom={year}&yearto={year}"
    else:
        url = f"https://openapi.naver.com/v1/search/movie.json?query={title}"

    res = requests.get(url, headers=header_parms).json()["items"][0]
    code = res["link"].split("=")[-1]
    scrapper = NaverReviewScrapper()
    story = scrapper.get_story(code=code)
    reviews = scrapper.get_review_by_num(title="", code=code)

    movie_info = {
        "title": re.sub("[^가-힣ㅏ-ㅣㄱ-ㅎ1-9\\s]", "", res["title"]),
        "subtitle": re.sub("[<b>|/]", "", res["subtitle"]),
        "image": get_movie_poster(
            f"https://movie.naver.com/movie/bi/mi/photoViewPopup.naver?movieCode={code}"
        ),
        "director": res["director"].split("|")[0],
        "actor": ", ".join(res["actor"].split("|")[:-1]),
        "rating": res["userRating"],
        "date": res["pubDate"],
        "story": story,
        "reviews": reviews,
        "code": code,
    }

    return movie_info


def movie_info_component(movie_info):
    st.subheader(f"{movie_info['title']} ({movie_info['date']})")
    st.caption(
        f"부제: {movie_info['subtitle'] if movie_info['subtitle'] else '정보가 없습니다.'}"
    )
    st.write(
        f"**감독**: {movie_info['director'] if movie_info['director'] else '정보가 없습니다.'}"
    )
    st.write(f"**배우**: {movie_info['actor'] if movie_info['actor'] else '정보가 없습니다.'}")
    st.write(f"**줄거리**:")
    st.info(movie_info["story"])


def app():
    tokenizer = load_tokenizer()
    model = load_model()
    st.title("네이버 영화 리뷰 분석")
    title = st.text_input("정확한 영화 제목을 입력하고 Enter를 눌러주세요. (시리즈 물은 번호를 포함해주세요.)")
    year = st.selectbox("개봉연도", make_year_option(), index=0)
    placeholder = st.empty()
    rating_area = st.empty()
    st.write("""---""")
    movie_aria = st.empty()

    if title:
        try:
            placeholder.empty()
            rating_area.empty()
            movie_aria.empty()
            placeholder.info("영화 내용을 최대한 빨리 정리하는 중... ")
            if year == "전체":
                movie_info = get_movie_info(title)
            else:
                movie_info = get_movie_info(title, year)
            if movie_info["reviews"]:
                placeholder.warning("열심히 리뷰를 읽고 분류 하는 중...")
                reviews = prediction(movie_info["reviews"], model, tokenizer)
            else:
                reviews = ""
            placeholder.info("페이지를 예쁘게 꾸미는 중...")
            sleep(0.8)
            placeholder.success("완료!")
            sleep(0.8)
            placeholder.empty()
            rating = movie_info["rating"]

            col3, col4, col5 = rating_area.columns(3)
            if not float(rating):
                col3.metric("평점", f"{rating}", "없음", delta_color="off")
                col4.metric("평가비율", "0%", "+ 긍정")
                col5.metric("평가비율", "0", "- 부정")
            elif float(rating) < 6:
                col3.metric("평점", f"{rating}", "- 부정", delta_color="normal")
            elif float(rating) < 7.5:
                col3.metric("평점", f"{rating}", "중립", delta_color="off")
            else:
                col3.metric("평점", f"{rating}", "+ 긍정", delta_color="normal")

            col1, col2 = movie_aria.columns([0.8, 1.5])
            with col1:
                poster = Image.open(
                    BytesIO(request.urlopen(movie_info["image"]).read()), mode="r"
                )
                st.image(poster)

            with col2:
                movie_info_component(movie_info)
                tab1, tab2, tab3 = st.tabs(["✅ 전체 평가", "😆 긍정 평가", "😡 부정 평가"])
                if not float(rating):
                    tab1.info(f"유저리뷰가 존재하지 않는 영화입니다.")
                    tab2.info(f"유저리뷰가 존재하지 않는 영화입니다.")
                    tab3.info(f"유저리뷰가 존재하지 않는 영화입니다.")
                else:
                    positive_reviews = reviews[reviews["감정"] == "긍정"]["리뷰"]
                    negative_reviews = reviews[reviews["감정"] == "부정"]["리뷰"]
                    col4.metric(
                        "평가비율",
                        f"{round(len(positive_reviews) / len(movie_info['reviews']) * 100)}%",
                        "+ 긍정",
                    )
                    col5.metric(
                        "평가비율",
                        f"{round(len(negative_reviews) / len(movie_info['reviews']) * 100)}%",
                        "- 부정",
                    )

                    tab1.subheader(f"전체 리뷰 ({len(reviews['리뷰'])}개)")
                    for review in reviews["리뷰"]:
                        tab1.code(review)

                    tab2.subheader(f"긍정 리뷰 ({len(positive_reviews)}개)")
                    for review in positive_reviews:
                        tab2.success(review)

                    tab3.subheader(f"부정 리뷰 ({len(negative_reviews)}개)")
                    for review in negative_reviews:
                        tab3.error(review)

        except Exception:
            placeholder.empty()
            st.error("존재하지 않는 영화입니다. 한번 더 확인해주세요.")
