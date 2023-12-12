from haystack.nodes import PreProcessor
from haystack.schema import Document

from typing import Optional, Literal, Tuple, List
import re

class RegexProcessor(PreProcessor):
    def __init__(self,
                 split_by = "passage",
                 split_respect_sentence_boundary = False,
                 split_regex = '.',
                 remove_substrings = ['\n\n'],
                 post_process = None,
                 add_page_number = False):
        super().__init__(split_by=split_by, split_respect_sentence_boundary=split_respect_sentence_boundary, remove_substrings=remove_substrings, split_length=1, split_overlap=0, add_page_number=add_page_number)

        self.split_regex = split_regex
        self._post_process = post_process

        if post_process:
            self._post_process = True
            self.post_process = post_process

        def _split_into_units(self, text:str, split_by: str):
            if split_by == "passage":
                elements = re.split(self.split_regex, text)
                split_at = '\n\n'
            elif split_by == "sentence":
                elements = self._split_sentences(text)
                split_at = ""
            elif split_by == "word":
                elements = text.split(" ")
                split_at = " "
            else:
                raise NotImplementedError("PreProcessor only supports 'passage', 'sentence', or 'word' split options")
            
            self._elements = elements
            return elements, split_at
        

def combine_chunks(docs: List[Document], parts:int = 2):
    combined_docs = []
    paras_per_chunk = len(docs)//parts
    extra_paras = len(docs)%parts

    start_index = 0
    for i in range(parts):
        end_index = start_index + paras_per_chunk
        if i < extra_paras:
            end_index += 1
        combined_doc = docs[start_index:end_index]
        content = ""
        for d in combined_doc:
            content += d.content
        combined_docs.append(content)

        start_index = end_index

        return combined_docs