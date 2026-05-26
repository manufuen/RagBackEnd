from langchain_text_splitters import RecursiveCharacterTextSplitter

''' 
Función para dividir el texto en fragmentos (chunks) 
'''
def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,     # Tamaño máximo de caracteres de cada chunk
        chunk_overlap=120,  # Solapamiento entre chunks para mantener contexto
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            ""
        ],
    )

    chunks = splitter.split_text(text)

    return [
        chunk.strip()
        for chunk in chunks
        if chunk and chunk.strip()
    ]