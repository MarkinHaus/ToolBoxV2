from langchain_core.documents import Document


def document_to_str_list(docs: list[Document]):
    final = []
    for doc in docs:
        final.append(doc.page_content)
