from langchain_core.documents import Document
from typing import List


def format_context_for_prompt(docs: List[Document]) -> str:
    """
    Builds a string context where each chunk is labeled for the LLM to cite.
    Matches the requirement: [Paper ID:chunk]
    """
    context_parts: List[str] = []
    
    for doc in docs:
        paper_id: str = doc.metadata.get("source", "Unknown")
        chunk_no: int = doc.metadata.get("chunk_id", 0)
        content: str = doc.page_content.strip().replace("\n", " ")
        
        # Create a clearly delimited block for the LLM
        header: str = f"--- [Paper ID:{paper_id}:chunk:{chunk_no}] ---"
        context_parts.append(f"{header}\n{content}")
        
    return "\n\n".join(context_parts)


# docs = [Document(metadata={'file_hash': '8bf4ea757c24d290502738bd2e48cf9432498121668bacceea576fa78b92fad6', 'chunk_id': 25, 'id': 'dea14a86-c09f-fc3b-1341-2e21ccd44492'}, page_content='counterparts. Our results suggest a non-Bloch framework fo r\nnon-Hermitian band topology.\nThere are many open questions ahead. For example, it\nis worthwhile to study the respective roles of the Bloch and\nnon-Bloch Chern numbers: What aspects of non-Hermitian\nphysics are described by the Bloch /non-Bloch one? In addi-\ntion, the theory can be generalized to many other topologica l\nnon-Hermitian systems. It is also interesting to go beyond t he\nband theory (e.g., to consider interaction e ﬀects).\nAcknowledgements.– W e would like to thank Hui Zhai for\ndiscussions. This work is supported by NSFC under grant No.\n11674189.\n∗ wangzhongemail@gmail.com\n[1] C. M. Bender, Reports on Progress in Physics 70, 947 (2007).\n[2] C. M. Bender and S. Boettcher, Physical Review Letters 80,\n5243 (1998).'),
#         Document(metadata={'file_hash': '57cc4e0c50f60fe30fce5d6350bd965900222f1ab200fa4dd273049bd426efb4', 'chunk_id': 6, 'id': '2667e9f2-dfde-a9b2-9984-a4912bf0082f'}, page_content='plicitly designed to satisfy cosmological and solar-syste m\nconstraints in certain limits of parameter space. W e use\nthese models to ask under what circumstances it is pos-\nsible to signiﬁcantly modify cosmological predictions and\nyet evade all local tests of gravity .\nW e begin in §II by introducing the model class, its ef-\nfect on the background expansion history and the growth\nof structure. W e show that cosmological tests of the\ngrowth of structure can, in principle, provide extremely\nprecise tests of f (R) gravity that rival local constraints'),
#         Document(metadata={'file_hash': 'bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697', 'chunk_id': 12, 'id': '85e3235c-3cd6-a6b0-712d-83c7d2351069'}, page_content='3.2 Attention\nAn attention function can be described as mapping a query and a set of key-value pairs to an output,\nwhere the query, keys, values, and output are all vectors. The output is computed as a weighted sum\n3'),
#         Document(metadata={'file_hash': 'bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697', 'chunk_id': 49, 'id': '780ffe2f-20b3-1585-1a60-293a7804b9e5'}, page_content='Attention Visualizations\nInput-Input Layer5\nIt\nis\nin\nthis\nspirit\nthat\na\nmajority\nof\nAmerican\ngovernments\nhave\npassed\nnew\nlaws\nsince\n2009\nmaking\nthe\nregistration\nor\nvoting\nprocess\nmore\ndifficult\n.\n<EOS>\n<pad>\n<pad>\n<pad>\n<pad>\n<pad>\n<pad>\nIt\nis\nin\nthis\nspirit\nthat\na\nmajority\nof\nAmerican\ngovernments\nhave\npassed\nnew\nlaws\nsince\n2009\nmaking\nthe\nregistration\nor\nvoting\nprocess\nmore\ndifficult\n.\n<EOS>\n<pad>\n<pad>\n<pad>\n<pad>\n<pad>\n<pad>\nFigure 3: An example of the attention mechanism following long-distance dependencies in the\nencoder self-attention in layer 5 of 6. Many of the attention heads attend to a distant dependency of\nthe verb ‘making’, completing the phrase ‘making...more difficult’. Attentions here shown only for\nthe word ‘making’. Different colors represent different heads. Best viewed in color.\n13'),
#         Document(metadata={'file_hash': 'bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697', 'chunk_id': 50, 'id': '3d291bda-1377-03fd-d5aa-af931f459a47'}, page_content='Input-Input Layer5\nThe\nLaw\nwill\nnever\nbe\nperfect\n,\nbut\nits\napplication\nshould\nbe\njust\n-\nthis\nis\nwhat\nwe\nare\nmissing\n,\nin\nmy\nopinion\n.\n<EOS>\n<pad>\nThe\nLaw\nwill\nnever\nbe\nperfect\n,\nbut\nits\napplication\nshould\nbe\njust\n-\nthis\nis\nwhat\nwe\nare\nmissing\n,\nin\nmy\nopinion\n.\n<EOS>\n<pad>\nInput-Input Layer5\nThe\nLaw\nwill\nnever\nbe\nperfect\n,\nbut\nits\napplication\nshould\nbe\njust\n-\nthis\nis\nwhat\nwe\nare\nmissing\n,\nin\nmy\nopinion\n.\n<EOS>\n<pad>\nThe\nLaw\nwill\nnever\nbe\nperfect\n,\nbut\nits\napplication\nshould\nbe\njust\n-\nthis\nis\nwhat\nwe\nare\nmissing\n,\nin\nmy\nopinion\n.\n<EOS>\n<pad>\nFigure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top:\nFull attentions for head 5. Bottom: Isolated attentions from just the word ‘its’ for attention heads 5\nand 6. Note that the attentions are very sharp for this word.\n14')]

# print(format_context_for_prompt(docs))