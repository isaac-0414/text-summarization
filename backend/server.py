from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor

def summarize_text(document):
  # Object of automatic summarization.
  auto_abstractor = AutoAbstractor()
  # Set tokenizer.
  auto_abstractor.tokenizable_doc = SimpleTokenizer()
  # Set delimiter for making a list of sentence.
  auto_abstractor.delimiter_list = [".", "\n"]
  # Object of abstracting and filtering document.
  abstractable_doc = TopNRankAbstractor()
  # Summarize document.
  result_dict = auto_abstractor.summarize(document, abstractable_doc)

  # Output result.
  return "".join(result_dict["summarize_result"])

class Item(BaseModel):
    text: str

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
   return {"name": "Isaac Z"}

@app.post("/summarize/")
def summarize(item: Item):
   print(type(item))
   textToSummarize = item.text
   summary = summarize_text(textToSummarize)
   return {"summary": summary}