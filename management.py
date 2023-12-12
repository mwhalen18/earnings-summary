from components.haystack_utils import RegexProcessor, combine_chunks
from components.prompt_utils import *

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PDFToTextConverter
from haystack.schema import Document
import openai
import os
from dotenv import load_dotenv
load_dotenv()


openai.api_base = os.getenv('OPENAI_API_BASE')
openai.organization = os.getenv('OPENAI_ORGANIZATION')
openai.api_key = os.getenv('OPENAI_API_KEY')

preprocessor = RegexProcessor(
    split_by='passage',
    split_regex='\n(?=[A-Z])',
    split_respect_sentence_boundary=False
)

file_path = 'data/STX_earnings_2023Q1.pdf'
#document_store = InMemoryDocumentStore(use_bm25=True, embedding_dim=384)
converter = PDFToTextConverter()

splits = ["at this time, we'll begin the question-and-answer session"]

doc = converter.convert(file_path)

for phrases in splits:
    if phrases in doc[0].content:
        mana_text = doc[0].content.split(phrases)
        qa_text = " ".join(doc[0].content.split(phrases)[1:])
        break
    else:
        print("no match")

mana = Document.from_dict({
    "content": mana_text[0],
    "meta": {"part": "Management Discussion Section"}
})

manager_discussion = preprocessor.process([mana])
manager_discussion_chunks = combine_chunks(manager_discussion)
print(manager_discussion_chunks[0])

primer = """
You are a detail-oriented, analytical assistant with expertise in financial analysis. Your task is to categorize and summarize information from a financial earnings call. Read the entire document and summarize the major themes/products/KPIs mentioned. For each major theme/product/KPI your summary should include the following in BRIEF bullet points:

Inlcude as many bullet points as is necessary to convey all of the relevant information.

Key Topic/KPI:
    - Key Numbers:
        *
    - Drivers of value:
        *
    - Additional Information:
        *

If any products or services are mentions, your summary should be much more detailed and include the following information:

    - Timelines:
    - Future plans and strategy:
        * 
    - Areas of excitiement:
        *
    - Challenges:
        * 
    - Associated costs:
        *
    - Projections:
        *
"""

nudge = "Dont forget about the special topics mentioned above"
advice = """
Notes:
Each bullet point should be no more than 10 words. Be brief bu be accurate
"""

def major_metrics_prompt(text):
    out = f"""
    CONTEXT = List the major financial metrics, topics, and products mentioned. If there is no additional information, write 'N/A' in the relevant section.

    DOCUEMENT: {text}
    """

    return out

response = openai.ChatCompletion.create(
    model = 'gpt-4',
    messages = [
        {'role': 'system', 'content': primer},
        {'role': 'system', 'content': nudge},
        {'role': 'user', 'content': major_metrics_prompt(manager_discussion_chunks[1])}
    ]
)

output = response['choices'][0]['message']['content']
output = output.replace('Key Topic/KPI: ', '')
output = output.replace('Product mentioned: ', '')

print(output)
