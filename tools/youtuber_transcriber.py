
from langchain.document_loaders import YoutubeLoader


def transcriber(list_urls):
    filestore = []
    count = 0
    for url in list_urls:
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,
            language=["en-US"],
            translation="en",
        )
        doc = loader.load()
        filestore.append(doc)
        with open(f'../data/video_{count}.txt', 'w', encoding='utf-8') as file:
            file.write(doc[0].page_content)
        count += 1

    return filestore


# urls = ["https://www.youtube.com/watch?v=9R8-NTqmm8g", "https://www.youtube.com/watch?v=oz8q2g2uQDE"]
# transcriber(urls)
