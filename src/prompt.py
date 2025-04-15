prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
If you detect any type of query has a specific language like Hindi, Marathi, you should be able to answer it in that particular language.
Helpful answer:
"""
