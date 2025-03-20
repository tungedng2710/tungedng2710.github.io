"""
Checkout this documents for installation: https://ai.google.dev/gemini-api/docs/quickstart?lang=python'
Refer link https://ai.google.dev/gemini-api/docs/api-key to get your Gemini API Key
For installation crawl4ai, please checkout this link: https://docs.crawl4ai.com/core/installation/
"""

import asyncio
import ast
from google import genai
# from google.genai import types
from crawl4ai import *
import PIL.Image

# Setup Gemini client
GEMINI_API_KEY = "ADD_YOUR_TOKEN_HERE"
GEMINI_MODEL_NAME = "gemini-2.0-flash"
CLIENT = genai.Client(api_key=GEMINI_API_KEY)


async def read_newspaper(url: str = None):
    """Read and summarize news articles from a given URL.
    Args:
        url (str, optional): _description_. Defaults to None.
    """
    assert url is not None, "URL is required"
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,
        )
        result_md = result.markdown
        response = CLIENT.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=["Lấy link 3 bài báo mới nhất, trả về dạng json với key là title và value là link \n\n", str(result_md)])
        
        result = response.text.replace("```json", "").replace("```", "")
        latest_posts = ast.literal_eval(result)
        for post_title in latest_posts:
            print('-'*50)
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(
                    url=latest_posts[post_title],
                )
                result_md = result.markdown
                response = CLIENT.models.generate_content(
                    model=GEMINI_MODEL_NAME,
                    contents=["Tóm tắt bài báo sau\n\n", str(result_md)])
                
            print(f"Title: {post_title}")
            print(f"Post URL: {latest_posts[post_title]}")
            print(f"Summarization: \n {response.text}")


def check_smoking(image_url: str = None):
    """Check if a person is smoking or not.
    Args:
        image_url (str, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    assert image_url is not None, "URL is required"
    image = PIL.Image.open(image_url)
    response = CLIENT.models.generate_content(
        model="gemini-2.0-flash",
        contents=["Is this person smoking? Confidence score?", image])
    print(response.text)
    return response.text


def check_using_mobile_phone(image_url: str = None):
    """Check if a person is using a mobile phone or not.
    Args:
        image_url (str, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    assert image_url is not None, "URL is required"
    image = PIL.Image.open(image_url)
    response = CLIENT.models.generate_content(
        model="gemini-2.0-flash",
        contents=["Is this person using a mobile phone? Confidence score?", image])
    print(response.text)
    return response.text
    

if __name__ == "__main__":
    # asyncio.run(read_newspaper(url = "https://goal.com"))
    # check_smoking(image_url="test.jpg")
    # check_using_mobile_phone(image_url="test.jpg")
    pass